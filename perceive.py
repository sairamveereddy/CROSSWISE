"""Frame-level perception for the demo crossings, from the YOLO11 boxes VAST stores per frame.

For every 0.2 s of video: track vehicles and pedestrians across frames, measure who is moving,
and decide a moment-level GO / WAIT / NO-GO so the robot reacts the instant the walk phase opens
instead of once per 5 s segment.

  NO-GO  a vehicle is moving through the near crosswalk zone
  GO     the walk phase is open: pedestrians are crossing in the zone, or the segment's signal read
         walk/green, AND no vehicle has moved through the zone for DEBOUNCE_S, AND the path is not blocked
  WAIT   anything else (unclear evidence is WAIT)

Output: data/perception.json
  {segment_id: {"zone": {v, p0, p1},
                "frames":  [{t, v: [[x1,y1,x2,y2, 2=moving through crossing|1=moving elsewhere|0=stopped, cls, dir]],
                             p: [[x1,y1,x2,y2, crossing]], l: [[x1,y1,x2,y2]]}],     # boxes normalised 0-1
                "moments": [{t, d, why, mv, xp}]}}                                   # 5 per second
Usage: python perceive.py [--debug]
"""
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import vss

HERE = pathlib.Path(__file__).resolve().parent
DECISIONS = HERE / "data" / "decisions.json"
OUT = HERE / "data" / "perception.json"

STEP = 6                      # 30 fps -> 5 fps
VEHICLES = {"car", "truck", "bus", "motorcycle"}
MIN_VEHICLE_H = 0.035         # ignore far-away vehicles (box height / frame height)
MOVING_V = 0.035              # frame-widths per second; slower = parked / queued
MOVING_P = 0.02
DEBOUNCE_S = 0.6              # zone must be clear of moving vehicles this long before GO
WALK_CONFIRM = 3              # consecutive 0.2 s samples with pedestrians crossing = walk phase open
PATH_HAZARDS = {"blocked_crosswalk", "construction"}
# Per fixed camera, calibrated by eye on annotated frames (fraction of frame height, box bottom edge):
#   vehicles whose bottom is below `v` are on the street the robot crosses;
#   pedestrians with bottom in [p0, p1] are on that crosswalk or its parallel twin (same signal phase).
ZONE = {"sf_streets_cam-4": {"v": 0.55, "p0": 0.45, "p1": 0.95},
        "sf_streets_cam-1": {"v": 0.54, "p0": 0.48, "p1": 0.80},
        "default": {"v": 0.55, "p0": 0.45, "p1": 0.95}}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def similarity(a, b):
    """IoU, or for small fast boxes that no longer overlap, closeness of centres relative to box width."""
    o = iou(a, b)
    if o >= 0.3:
        return o
    w = max(a[2] - a[0], b[2] - b[0])
    d = abs((a[0] + a[2]) - (b[0] + b[2])) / 2 + abs((a[1] + a[3]) - (b[1] + b[3])) / 2
    return 0.29 * max(0.0, 1 - d / (0.9 * w)) if w else 0.0


def track(frames, W, H, dt, prev=None):
    """Greedy matching between consecutive sampled frames; returns per-frame boxes with velocity."""
    prev = prev or {"v": [], "p": []}
    out = []
    for f in frames:
        cur = {"v": [], "p": [], "l": []}
        for d in f["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            box = [x1 / W, y1 / H, x2 / W, y2 / H]
            if d["label"] in VEHICLES and (box[3] - box[1]) >= MIN_VEHICLE_H and d["confidence"] >= 0.35:
                cur["v"].append({"b": box, "c": d["label"]})
            elif d["label"] == "person" and d["confidence"] >= 0.4:
                cur["p"].append({"b": box, "c": "person"})
            elif d["label"] == "traffic light" and d["confidence"] >= 0.3:
                cur["l"].append({"b": box})
        for k in ("v", "p"):
            used = set()
            for o in cur[k]:
                best, bi = 0.05, None
                for i, q in enumerate(prev[k]):
                    s = similarity(o["b"], q["b"])
                    if i not in used and s > best:
                        best, bi = s, i
                if bi is None:
                    o["vx"] = o["vy"] = 0.0
                    o["age"] = 0
                    continue
                used.add(bi)
                q = prev[k][bi]
                cx, cy = (o["b"][0] + o["b"][2]) / 2, (o["b"][1] + o["b"][3]) / 2
                px, py = (q["b"][0] + q["b"][2]) / 2, (q["b"][1] + q["b"][3]) / 2
                # exponential smoothing kills YOLO box jitter on parked cars
                o["vx"] = 0.5 * q["vx"] + 0.5 * (cx - px) / dt
                o["vy"] = 0.5 * q["vy"] + 0.5 * (cy - py) / dt * H / W
                o["age"] = q["age"] + 1
        out.append(cur)
        prev = cur
    return out


def speed(o):
    return (o["vx"] ** 2 + o["vy"] ** 2) ** 0.5


def r3(b):
    return [round(x, 3) for x in b]


def analyse_clip(segs, dets):
    """Process one clip's segments in order so tracks (and the walk/danger timers) carry across segments."""
    res, prev = {}, None
    last_danger, walk_run = -99.0, 0
    for seg, det in zip(segs, dets):
        H, W = det["video_shape"][:2]
        frames = det["frames"][::STEP]
        tracked = track(frames, W, H, STEP / (det.get("fps") or 30.0), prev)
        prev = tracked[-1] if tracked else prev
        z = ZONE.get(seg["camera_id"], ZONE["default"])
        walk_signal = seg.get("signal") == "walk"
        blocked = bool(PATH_HAZARDS & set(seg.get("hazards") or [])) or (seg.get("cosmos_tags") or {}).get("blocked") == "yes"
        out_frames, moments = [], []
        for f, cur in zip(frames, tracked):
            t = round(seg["start_sec"] + f["time_sec"], 2)
            moving = [o for o in cur["v"] if o["age"] >= 1 and speed(o) >= MOVING_V]
            danger = [o for o in moving if o["b"][3] >= z["v"]]
            xp = [o for o in cur["p"] if o["age"] >= 1 and z["p0"] <= o["b"][3] <= z["p1"] and speed(o) >= MOVING_P]
            if danger:
                last_danger = t
            walk_run = walk_run + 1 if xp else 0
            walk_phase = walk_run >= WALK_CONFIRM or walk_signal
            clear_for = t - last_danger
            n = len(danger)
            if danger:
                d, why = "NO-GO", f"{n} vehicle{'s' if n > 1 else ''} moving through the crossing"
            elif blocked:
                d, why = "WAIT", "crosswalk obstructed"
            elif walk_phase and clear_for >= DEBOUNCE_S:
                d, why = "GO", (f"walk phase open: {len(xp)} pedestrian{'s' if len(xp) != 1 else ''} crossing, traffic stopped"
                                if xp else "walk signal, traffic stopped")
            elif walk_phase:
                d, why = "WAIT", "walk phase open, confirming traffic has stopped"
            elif xp:
                d, why = "WAIT", "pedestrians stepping out, confirming walk phase"
            else:
                d, why = "WAIT", "no walk phase yet: nobody crossing"
            moments.append({"t": t, "d": d, "why": why, "mv": n, "xp": len(xp)})
            out_frames.append({"t": t,
                               "v": [r3(o["b"]) + [int(o in danger) * 2 or int(o in moving), o["c"], 1 if o["vx"] >= 0 else -1]
                                     for o in cur["v"]],
                               "p": [r3(o["b"]) + [int(o in xp)] for o in cur["p"]],
                               "l": [r3(o["b"]) for o in cur["l"]]})
        res[seg["segment_id"]] = {"zone": z, "frames": out_frames, "moments": moments}
    return res


def main(debug=False):
    rows = [r for r in json.loads(DECISIONS.read_text()) if r["caption_source"] != "vss_index"]
    with ThreadPoolExecutor(6) as ex:
        dets = dict(zip([r["segment_id"] for r in rows],
                        ex.map(lambda r: vss.get("/videos/detections", source=r["segment_id"]), rows)))
    clips = {}
    for r in rows:
        clips.setdefault(r["original_video"], []).append(r)
    res = {}
    for segs in clips.values():
        segs.sort(key=lambda x: x["segment_number"])
        res.update(analyse_clip(segs, [dets[s["segment_id"]] for s in segs]))
    OUT.write_text(json.dumps(res, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(res)} segments)")
    by_video = {}
    for r in rows:
        by_video.setdefault(r["original_video"], []).append(r)
    for v, segs in by_video.items():
        print("\n" + v.rsplit("/", 1)[-1])
        for s in sorted(segs, key=lambda x: x["segment_number"]):
            ms = res[s["segment_id"]]["moments"]
            strip = "".join({"GO": "G", "WAIT": ".", "NO-GO": "X"}[m["d"]] for m in ms)
            print(f"  seg {s['segment_number']} LLM={s['decision']:5s} sig={s['signal']:9s} {strip}"
                  + (f"  mv={[m['mv'] for m in ms]} xp={[m['xp'] for m in ms]}" if debug else ""))


if __name__ == "__main__":
    main(debug="--debug" in sys.argv)
