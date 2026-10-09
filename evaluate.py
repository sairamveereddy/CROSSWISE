"""CrossWise evaluation against human safe/unsafe labels.

  python evaluate.py select              -> data/label_set.json (20 segments, stratified, deterministic)
  python evaluate.py score [labels.json] -> data/eval_results.json + a W&B run
       labels default: fetched from the deployed app (GET /app/api/labels), else data/labels.json

Mapping: GO = predicted safe to start crossing; WAIT and NO-GO = predicted not safe now.
"Unsafe GO" (GO on a segment the human labelled unsafe) is the critical error and is reported separately.
"""
import collections
import json
import os
import pathlib
import random
import sys
import urllib.request

DATA = pathlib.Path(__file__).resolve().parent / "data"
APP_LABELS_URL = "http://video-lab-team-48.cosmos.vastdata.com/app/api/labels"
QUOTA = {"GO": 8, "WAIT": 6, "NO-GO": 6}


def select(seed=48):
    rows = json.loads((DATA / "decisions.json").read_text())
    rng = random.Random(seed)
    picked = []
    for decision, k in QUOTA.items():
        pool = [r for r in rows if r["decision"] == decision]
        demo = [r for r in pool if r["caption_source"] != "vss_index"]
        rest = [r for r in pool if r["caption_source"] == "vss_index"]
        rng.shuffle(demo)
        rng.shuffle(rest)
        # round-robin across cameras so no single crossing dominates
        by_cam = collections.defaultdict(list)
        for r in demo[:2] + rest:
            by_cam[r["camera_id"]].append(r)
        chosen = []
        while len(chosen) < k and any(by_cam.values()):
            for cam in sorted(by_cam):
                if by_cam[cam] and len(chosen) < k:
                    chosen.append(by_cam[cam].pop(0))
        picked += chosen
    rng.shuffle(picked)
    out = [{"item": i + 1, "segment_id": r["segment_id"], "camera_id": r["camera_id"],
            "start_sec": r["start_sec"], "end_sec": r["end_sec"]} for i, r in enumerate(picked)]
    (DATA / "label_set.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} items:", collections.Counter(r["decision"] for r in picked),
          collections.Counter(r["camera_id"] for r in picked))


def load_labels(path=None):
    if path:
        return json.loads(pathlib.Path(path).read_text())
    try:
        with urllib.request.urlopen(APP_LABELS_URL, timeout=20) as r:
            labels = json.load(r)
        (DATA / "labels.json").write_text(json.dumps(labels, indent=1))
        return labels
    except Exception as e:
        print("could not fetch labels from app:", e)
        return json.loads((DATA / "labels.json").read_text())


def score(labels):
    rows = {r["segment_id"]: r for r in json.loads((DATA / "decisions.json").read_text())}
    items = []
    for seg, lab in labels.items():
        lab = lab["label"] if isinstance(lab, dict) else lab
        if lab not in ("safe", "unsafe") or seg not in rows:
            continue
        r = rows[seg]
        items.append({"segment_id": seg, "camera_id": r["camera_id"], "start_sec": r["start_sec"], "human": lab,
                      "decision": r["decision"], "llm_decision": r["llm_decision"], "unclear": r["unclear"],
                      "caption_source": r["caption_source"], "reasons": r["reasons"]})
    n = len(items)
    pred_safe = lambda d: d == "GO"
    agree = sum(pred_safe(i["decision"]) == (i["human"] == "safe") for i in items)
    agree_llm = sum(pred_safe(i["llm_decision"]) == (i["human"] == "safe") for i in items)
    confusion = collections.Counter((i["human"], i["decision"]) for i in items)
    unsafe_go = [i for i in items if i["decision"] == "GO" and i["human"] == "unsafe"]
    unsafe_go_llm = [i for i in items if i["llm_decision"] == "GO" and i["human"] == "unsafe"]
    missed_go = [i for i in items if i["decision"] != "GO" and i["human"] == "safe"]
    gos = sum(i["decision"] == "GO" for i in items)
    res = {"n_labelled": n, "agreement": round(agree / n, 3) if n else None,
           "agreement_without_guardrail": round(agree_llm / n, 3) if n else None,
           "go_predictions": gos, "unsafe_go_count": len(unsafe_go),
           "unsafe_go_rate_of_gos": round(len(unsafe_go) / gos, 3) if gos else None,
           "unsafe_go_count_without_guardrail": len(unsafe_go_llm),
           "missed_safe_count": len(missed_go),
           "confusion": {f"human={h}|decision={d}": c for (h, d), c in sorted(confusion.items())},
           "unsafe_go": unsafe_go, "missed_safe": missed_go, "items": items}
    (DATA / "eval_results.json").write_text(json.dumps(res, indent=1))
    return res


def log_wandb(res):
    import wandb
    run = wandb.init(entity=os.environ["WANDB_TEAM"], project=os.environ["WANDB_PROJECT"], job_type="eval",
                     name="crosswise-human-eval", config={"mapping": "GO=safe; WAIT/NO-GO=not safe", "quota": QUOTA})
    run.log({k: v for k, v in res.items() if isinstance(v, (int, float)) and v is not None})
    cols = ["segment_id", "camera_id", "start_sec", "human", "decision", "llm_decision", "unclear", "caption_source", "reasons"]
    run.log({"labelled_segments": wandb.Table(columns=cols, data=[[i[c] if c != "reasons" else " / ".join(i[c])
                                                                   for c in cols] for i in res["items"]]),
             "unsafe_go": wandb.Table(columns=cols, data=[[i[c] if c != "reasons" else " / ".join(i[c])
                                                           for c in cols] for i in res["unsafe_go"]])})
    run.summary.update({k: v for k, v in res.items() if k in ("agreement", "unsafe_go_count", "n_labelled",
                                                            "agreement_without_guardrail", "missed_safe_count")})
    url = run.url
    run.finish()
    return url


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "select":
        select()
    else:
        res = score(load_labels(sys.argv[2] if len(sys.argv) > 2 else None))
        print(json.dumps({k: v for k, v in res.items() if k not in ("items", "unsafe_go", "missed_safe")}, indent=1))
        for i in res["unsafe_go"]:
            print("UNSAFE GO:", i["camera_id"], i["start_sec"], i["reasons"])
        if "--no-wandb" not in sys.argv:
            print("W&B run:", log_wandb(res))
