"""CrossWise decision engine: caption + YOLO counts -> GO / WAIT / NO-GO with reasons.

Decision support on recorded footage only; it does not control a robot.

Inputs
  data/cosmos_captions.json   demo segments captioned by Cosmos Reason with the CrossWise prompt
  VSS index (/tools/segments) a per-camera sample with the original index captions (scorecard)
Outputs
  data/decisions.json                     one row per segment (also bundled into the web app)
  VastDB  <VASTDB_BUCKET>/hackathon/crossing_decisions   same rows, appended per run
  Weave   <WANDB_TEAM>/<WANDB_PROJECT>    one trace per decision call

Usage: python decide.py [--sample-per-camera N] [--model MODEL] [--no-vastdb] [--demo-only]
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import random
import uuid
from concurrent.futures import ThreadPoolExecutor

import openai
import weave

import vss

DATA = pathlib.Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
SCORECARD_CAMERAS = ["sf_streets_cam-1", "sf_streets_cam-2", "sf_streets_cam-3", "sf_streets_cam-4",
                     "sf_streets_cam-5", "nyc_streets_cam-1", "nyc_streets_cam-2"]
VEHICLES = ("car", "truck", "bus", "motorcycle")
HAZARDS = ["turning_vehicle", "vehicle_in_crosswalk", "moving_traffic", "blocked_crosswalk", "construction",
           "emergency_vehicle", "cyclists", "crowded_crosswalk", "signal_not_visible", "low_visibility"]

SYSTEM = f"""You are CrossWise, a street-crossing decision aid for a sidewalk delivery robot waiting at the curb.
You get one 5-second video segment described by a vision-language model caption plus YOLO object counts.
Decide whether the robot should start crossing now.

Decisions:
- GO: evidence clearly supports crossing now: a walk signal, or a green light with pedestrians crossing,
  AND no vehicle moving or turning through the crosswalk, AND the crosswalk/curb is passable.
- WAIT: not now, but likely to clear soon: red light or don't-walk with no immediate danger, traffic passing
  across the road, pedestrians crowding the curb, or a PARTIAL obstruction that still leaves a passable path.
- NO-GO: immediate danger or no passable path right now: a vehicle actively turning or driving through the
  crosswalk the robot would use, an emergency vehicle passing, or the crosswalk/curb FULLY blocked.
  A signal alone (red, don't-walk) is WAIT, not NO-GO.
If the evidence is ambiguous or contradictory, answer WAIT with "unclear": true. Never GO on unclear evidence.
List only hazards the evidence actually shows (stopped vehicles are not "moving_traffic").
A "traffic light" colour without a pedestrian signal may be the vehicle signal facing the camera; treat it as weaker
evidence than a walk / don't-walk signal.

Never describe or identify individual people (no clothing, age, gender, appearance) and never mention licence plates.
Refer to people only as "pedestrians" or "cyclists".

Reply with JSON only:
{{"decision": "GO"|"WAIT"|"NO-GO", "unclear": bool, "signal": "walk"|"dont_walk"|"green"|"red"|"yellow"|"none"|"unclear",
 "hazards": [subset of {HAZARDS}], "reasons": [1-3 short reasons, max 15 words each], "confidence": 0.0-1.0}}"""


def yolo_summary(counts):
    counts = counts if isinstance(counts, dict) else json.loads(counts or "{}")
    keep = {k: v for k, v in counts.items() if k in VEHICLES + ("person", "bicycle", "traffic light", "stop sign")}
    keep["vehicles_total"] = sum(counts.get(k, 0) for k in VEHICLES)
    return keep


def guardrail(result, tags):
    """Hard safety rule on top of the LLM: never let a GO stand against explicit hazard tags from Cosmos."""
    reasons = list(result.get("reasons") or [])
    decision = result.get("decision")
    if decision == "GO" and (tags.get("vehicles") == "turning" or tags.get("blocked") == "yes"):
        what = "turning vehicle" if tags.get("vehicles") == "turning" else "blocked crosswalk"
        return "WAIT", reasons + [f"Guardrail: Cosmos flagged a {what}; GO downgraded"]
    if decision == "GO" and result.get("unclear"):
        return "WAIT", reasons + ["Guardrail: evidence unclear; GO downgraded"]
    return decision, reasons


class Engine:
    def __init__(self, model):
        self.model = model
        self.client = openai.OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"],
                                    project=f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}")

    @weave.op()
    def decide(self, segment_id: str, camera_id: str, caption: str, yolo: dict, cosmos_tags: dict) -> dict:
        user = (f"Camera: {camera_id}\nCaption:\n{caption}\n\nYOLO object counts over the segment: {json.dumps(yolo)}\n"
                f"Structured tags from the caption (may be missing or wrong): {json.dumps(cosmos_tags) or '{}'}")
        last_err = None
        for _ in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=400, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
                out = json.loads(r.choices[0].message.content)
                break
            except Exception as e:  # malformed JSON or transient API error: retry
                last_err = e
        else:
            out = {"decision": "WAIT", "unclear": True, "signal": "unclear", "hazards": [],
                   "reasons": [f"Decision engine error; defaulting to WAIT ({type(last_err).__name__})"],
                   "confidence": 0.0}
        out["decision"] = str(out.get("decision", "WAIT")).upper().replace("NOGO", "NO-GO").replace("NO_GO", "NO-GO")
        if out["decision"] not in ("GO", "WAIT", "NO-GO"):
            out.update(decision="WAIT", unclear=True)
        out["hazards"] = [h for h in out.get("hazards") or [] if h in HAZARDS]
        out["llm_decision"] = out["decision"]
        out["decision"], out["reasons"] = guardrail(out, cosmos_tags or {})
        return out


def demo_segments():
    rows = json.loads((DATA / "cosmos_captions.json").read_text())
    return [{**r, "caption_source": "cosmos_reason_direct"} for r in rows]


def sample_segments(per_camera, seed=48):
    """Whole consecutive chunks per camera so wait-before-GO streaks are meaningful."""
    chunks, offset = [], 0
    while True:
        r = vss.get("/videos/explore", scope="all", limit=100, offset=offset)
        chunks += r["chunks"]
        offset += 100
        if offset >= r["total"]:
            break
    demo = {r["original_video"] for r in demo_segments()}
    rng = random.Random(seed)
    out = []
    for cam in SCORECARD_CAMERAS:
        parents = sorted({c["original_video"] for c in chunks if c.get("camera_id") == cam} - demo)
        for pv in rng.sample(parents, min(len(parents), max(1, per_camera // 6))):
            for s in vss.get("/tools/segments", original_video=pv)["segments"]:
                out.append({"original_video": pv, "source": s["source"], "camera_id": cam, "location": s.get("location"),
                            "segment_number": s.get("segment_number"), "start_sec": s.get("segment_start_sec"),
                            "end_sec": s.get("segment_end_sec"), "object_counts": s.get("object_counts"),
                            "caption": s.get("reasoning_content") or "", "tags": {}, "caption_source": "vss_index"})
    return out


def run(segments, engine, run_id):
    def one(seg):
        res = engine.decide(seg["source"], seg["camera_id"], seg["caption"], yolo_summary(seg.get("object_counts")),
                            seg.get("tags") or {})
        return {"run_id": run_id, "segment_id": seg["source"], "original_video": seg["original_video"],
                "camera_id": seg["camera_id"], "location": seg.get("location") or "",
                "segment_number": int(seg.get("segment_number") or 0), "start_sec": float(seg.get("start_sec") or 0),
                "end_sec": float(seg.get("end_sec") or 0), "caption_source": seg["caption_source"],
                "cosmos_tags": seg.get("tags") or {}, "yolo": yolo_summary(seg.get("object_counts")),
                "decision": res["decision"], "llm_decision": res["llm_decision"], "unclear": bool(res.get("unclear")),
                "signal": res.get("signal", "unclear"), "hazards": res["hazards"], "reasons": res["reasons"],
                "confidence": float(res.get("confidence") or 0), "model": engine.model,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, segments))


def write_vastdb(rows):
    import pyarrow as pa
    import vastdb
    cols = pa.schema([
        ("run_id", pa.utf8()), ("segment_id", pa.utf8()), ("original_video", pa.utf8()), ("camera_id", pa.utf8()),
        ("location", pa.utf8()), ("segment_number", pa.int32()), ("start_sec", pa.float32()), ("end_sec", pa.float32()),
        ("caption_source", pa.utf8()), ("cosmos_tags", pa.utf8()), ("yolo_counts", pa.utf8()), ("decision", pa.utf8()),
        ("llm_decision", pa.utf8()), ("unclear", pa.bool_()), ("signal", pa.utf8()), ("hazards", pa.utf8()),
        ("reasons", pa.utf8()), ("confidence", pa.float32()), ("model", pa.utf8()), ("created_at", pa.utf8())])
    data = {c: [] for c in cols.names}
    for r in rows:
        for c in cols.names:
            v = r["yolo"] if c == "yolo_counts" else r[c]
            data[c].append(json.dumps(v) if isinstance(v, (dict, list)) else v)
    session = vastdb.connect(endpoint=os.environ["S3_ENDPOINT"], access=os.environ["ACCESS_KEY"],
                             secret=os.environ["SECRET_KEY"], ssl_verify=False)
    with session.transaction() as tx:
        bucket = tx.bucket(os.environ["VASTDB_BUCKET"])
        schema = bucket.schema("hackathon", fail_if_missing=False) or bucket.create_schema("hackathon", fail_if_exists=False)
        table = schema.table("crossing_decisions", fail_if_missing=False) or \
            schema.create_table("crossing_decisions", columns=cols, fail_if_exists=False)
        table.insert(pa.table(data, schema=cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sample-per-camera", type=int, default=30)
    ap.add_argument("--demo-only", action="store_true")
    ap.add_argument("--no-vastdb", action="store_true")
    ap.add_argument("--videos", nargs="+", help="decide only these captioned clips and merge into decisions.json")
    args = ap.parse_args()

    weave.init(f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}")
    if args.videos:
        segments = [s for s in demo_segments() if s["original_video"] in set(args.videos)]
    else:
        segments = demo_segments() + ([] if args.demo_only else sample_segments(args.sample_per_camera))
    run_id = f"run-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    print(f"{run_id}: deciding {len(segments)} segments with {args.model}")
    rows = run(segments, Engine(args.model), run_id)
    if args.videos:
        new = {r["segment_id"] for r in rows}
        old = [r for r in json.loads((DATA / "decisions.json").read_text()) if r["segment_id"] not in new]
        (DATA / "decisions.json").write_text(json.dumps(old + rows, indent=1))
    else:
        (DATA / "decisions.json").write_text(json.dumps(rows, indent=1))
    if not args.no_vastdb:
        write_vastdb(rows)
        print(f"wrote {len(rows)} rows to {os.environ['VASTDB_BUCKET']}/hackathon/crossing_decisions")
    by = {}
    for r in rows:
        by.setdefault(r["camera_id"], []).append(r["decision"])
    for cam, ds in by.items():
        print(f"  {cam:18s} n={len(ds):3d} GO={ds.count('GO'):3d} WAIT={ds.count('WAIT'):3d} NO-GO={ds.count('NO-GO'):3d}")


if __name__ == "__main__":
    main()
