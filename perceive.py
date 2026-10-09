"""Moment-level crossing perception for the demo clips (5 decisions per second).

Inputs per 5 s segment:
  - YOLO11 boxes VAST stores for every frame (/videos/detections)          -> vehicle + pedestrian tracks
  - the segment video (H.264, /videos/stream)                              -> pedestrian-signal colour (walk / hand)
  - one mid-segment frame sent to Gemma 4 on W&B Inference (CoreWeave)     -> obstacles on the crosswalk + free path
  - the segment's Cosmos tags                                              -> blocked-crosswalk fallback

Policy every 0.2 s:
  NO-GO  a vehicle is moving through the crossing zone, or is predicted to enter it within PREDICT_S
  WAIT   pedestrian signal shows the hand (steady or flashing countdown), or the free path is too narrow,
         or traffic is heavy and the zone has not been quiet long enough
  GO     walk signal (when a pedestrian signal is visible) or a clear gap (when none is), with no conflict now or predicted

Output: data/perception.json
  {segment_id: {"cfg": {...}, "obstacles": {...},
                "frames":  [{t, v: [[x1,y1,x2,y2, 2=conflict|1=moving|0=stopped, cls, vx, vy]], p: [[x1,y1,x2,y2, crossing]],
                             h: [x1,y1,x2,y2, reading] | null}],                      # boxes normalised 0-1, v in frame/s
                "moments": [{t, d, why, sig, mv, pred, xp, lvl, quiet}]}}
Usage: python perceive.py [--debug] [--refresh-obstacles]
"""
import base64
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

import vss

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = pathlib.Path(os.environ.get("CROSSWISE_CACHE", HERE / "cache"))
OUT = DATA / "perception.json"
OBSTACLES = DATA / "obstacles.json"

STEP = 6                       # 30 fps -> 5 fps
VEHICLES = {"car", "truck", "bus", "motorcycle"}
MIN_VEHICLE_H = 0.035          # ignore far-away vehicles (box height / frame height)
MOVING_V = 0.035               # frame-widths per second; slower = parked / queued
MOVING_P = 0.02
MOTION_N = 3                   # velocity averaged over this many 0.2 s steps
PREDICT_S = 1.5                # look-ahead for vehicles about to enter the crossing zone
QUIET_LIGHT, QUIET_HEAVY = 0.6, 2.0   # seconds the zone must be quiet before GO without a walk signal
HEAVY_SHARE = 0.30             # share of the last 10 s with a conflict that counts as heavy traffic
WALK_CONFIRM = 3               # consecutive samples of the walking-person lamp before GO (0.6 s)
WALK_CONFIRM_REMEMBERED = 6    # same, when YOLO missed the head and its last known position is read (1.2 s)
SIGNAL_HOLD_S = 2.0            # keep the last signal reading while a pedestrian blocks the view
STEADY_S = 3.0                 # hand lit this long without blinking off = countdown over, steady don't-walk
ROBOT_MPS = 1.6
PATH_HAZARDS = {"blocked_crosswalk", "construction"}
VISION_MODEL = "google/gemma-4-31B-it"

# Per fixed camera, calibrated on annotated frames. v: vehicles with box bottom below this line are on the street the
# robot crosses. p0..p1: pedestrians in this band are on that crosswalk or its parallel twin. width_m: crossing length.
# x0..x1: horizontal extent of the crossing corridor (vehicles outside it do not conflict).
# head: x-range of the pedestrian signal that faces the robot across ITS crosswalk; other heads in view govern other
# crosswalks and are ignored (None = no readable head, gap acceptance only).
CAMERAS = {"sf_streets_cam-4": {"v": 0.55, "p0": 0.45, "p1": 0.95, "x0": 0.0, "x1": 1.0, "width_m": 16, "head": None},
           "sf_streets_cam-1": {"v": 0.54, "p0": 0.48, "p1": 0.80, "x0": 0.0, "x1": 0.48, "width_m": 12, "head": None},
           "nyc_streets_cam-1": {"v": 0.55, "p0": 0.45, "p1": 0.97, "x0": 0.05, "x1": 1.0, "width_m": 11,
                                 "head": [0.0, 0.2]},
           "default": {"v": 0.55, "p0": 0.45, "p1": 0.95, "x0": 0.0, "x1": 1.0, "width_m": 12, "head": [0.0, 1.0]}}


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


def camera_shift(cur, prev):
    """Image shift between samples caused by the camera itself: signal heads never move, and most vehicles in a
    queue are parked or stopped, so the median displacement of matched static-ish boxes is the camera motion."""
    shifts = []
    for k in ("l", "v"):
        for o in cur.get(k, []):
            best, bq = 0.3, None
            for q in prev.get(k, []):
                s = iou(o["b"], q["b"])
                if s > best:
                    best, bq = s, q
            if bq:
                shifts.append(((o["b"][0] + o["b"][2] - bq["b"][0] - bq["b"][2]) / 2,
                               (o["b"][1] + o["b"][3] - bq["b"][1] - bq["b"][3]) / 2, k))
        if k == "l" and len(shifts) >= 1:
            break
    if not shifts or (shifts[0][2] == "v" and len(shifts) < 3):
        return 0.0, 0.0
    return float(np.median([s[0] for s in shifts])), float(np.median([s[1] for s in shifts]))


def edge_shift(a0, a1, b0, b1):
    """Displacement along one axis from box edges (a = now, b = before). A real move shifts both edges the same way;
    a person walking in front of a parked car, or the frame border cutting the box, moves only one edge. Edges on the
    frame border are ignored; if the two edges disagree there is no motion."""
    d = []
    if a0 > 0.005 and b0 > 0.005:
        d.append(a0 - b0)
    if a1 < 0.995 and b1 < 0.995:
        d.append(a1 - b1)
    if len(d) == 2:
        return 0.0 if d[0] * d[1] <= 0 else min(d, key=abs)
    return d[0] if d else 0.0


def steady(d):
    """Mean step, or 0 when the steps change direction (box jitter on a stopped vehicle)."""
    return 0.0 if min(d) < 0 < max(d) else sum(d) / len(d)


def track(frames, W, H, dt, prev=None):
    """Greedy matching between consecutive sampled frames; returns per-frame boxes with smoothed velocity."""
    prev = prev or {"v": [], "p": [], "l": []}
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
                cur["l"].append({"b": box, "px": d["bbox"]})
        cam_dx, cam_dy = camera_shift(cur, prev)
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
                    o["age"], o["hist"] = 0, []
                    continue
                used.add(bi)
                q, a = prev[k][bi], o["b"]
                b = [q["b"][0] + cam_dx, q["b"][1] + cam_dy, q["b"][2] + cam_dx, q["b"][3] + cam_dy]
                # camera motion removed (handheld footage); velocity = mean over the last MOTION_N steps
                o["hist"] = (q["hist"] + [(edge_shift(a[0], a[2], b[0], b[2]), edge_shift(a[1], a[3], b[1], b[3]))])[-MOTION_N:]
                o["vx"], o["vy"] = (steady([h[i] for h in o["hist"]]) / dt * s for i, s in ((0, 1), (1, H / W)))
                o["age"] = q["age"] + 1
        out.append(cur)
        prev = cur
    return out


def speed(o):
    return (o["vx"] ** 2 + o["vy"] ** 2) ** 0.5


def r3(b):
    return [round(x, 3) for x in b]


# ---------- video + pedestrian signal ----------

def video_path(seg_id):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / seg_id.rsplit("/", 1)[-1]
    if not p.exists():
        r = requests.get(f"{vss.BASE}/videos/stream", params={"source": seg_id, "token": vss.login()}, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def detections(seg_id):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (seg_id.rsplit("/", 1)[-1] + ".det.json")
    if p.exists():
        return json.loads(p.read_text())
    for attempt in range(4):
        try:
            det = vss.get("/videos/detections", source=seg_id)
            p.write_text(json.dumps(det))
            return det
        except requests.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def decode(path, W, H, fps=5):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    return np.frombuffer(raw, np.uint8)[: n * W * H * 3].reshape(n, H, W, 3)


def read_head(img, px):
    """Pedestrian signal heads are near-square boxes; the left lamp shows the hand (red/orange) or walking person (white)."""
    x1, y1, x2, y2 = px
    if (y2 - y1) < 45 or not 0.7 <= (y2 - y1) / max(1, x2 - x1) <= 1.45:
        return None
    w, h = x2 - x1, y2 - y1
    # inside of the left lamp only: the yellow-orange housing must not look like a lit hand
    c = img[y1 + h * 15 // 100: y2 - h * 15 // 100, x1 + w * 12 // 100: x1 + w * 58 // 100].astype(np.float32) / 255
    if c.size == 0:
        return None
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1)
    s = (mx - mn) / (mx + 1e-6)
    hand = ((r > .55) & (g < r * .62) & (b < r * .55) & (s > .5)).mean()
    walk = ((mn > .7) & (s < .22)).mean()
    if hand > .025 and hand > walk:
        return "hand"
    if walk > .02:
        return "walk"
    return "dark"


# ---------- obstacles on the crosswalk (Gemma 4 on W&B Inference) ----------

OBSTACLE_PROMPT = ("This is a street camera frame. A small sidewalk delivery robot must cross the road on the crosswalk "
                   "nearest the camera. Return JSON only: {\"obstacles\":[{\"type\":\"barrier|cone|stopped_vehicle|other\","
                   "\"x0\":0-1,\"x1\":0-1}],\"free_x0\":0-1,\"free_x1\":0-1,\"passable\":true|false}. x values are fractions "
                   "of image width for objects ON that crosswalk (ignore people); free_x0..free_x1 is the widest clear "
                   "horizontal span of the crosswalk. Never describe people.")


def look_for_obstacles(path, client):
    jpg = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "2.5", "-i", str(path), "-frames:v", "1",
                          "-vf", "scale=1280:-1", "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True, check=True).stdout
    r = client.chat.completions.create(model=VISION_MODEL, temperature=0, max_tokens=300, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
        {"type": "text", "text": OBSTACLE_PROMPT}]}])
    txt = r.choices[0].message.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    out = json.loads(txt)
    out["obstacles"] = [o for o in out.get("obstacles", []) if o.get("type") != "other" or o.get("x1", 0) > o.get("x0", 0)]
    return {"obstacles": out["obstacles"], "free": [float(out.get("free_x0", 0)), float(out.get("free_x1", 1))],
            "passable": bool(out.get("passable", True)), "model": VISION_MODEL}


def obstacles_for(rows, refresh=False):
    cache = json.loads(OBSTACLES.read_text()) if OBSTACLES.exists() and not refresh else {}
    todo = [r for r in rows if r["segment_id"] not in cache]
    if todo:
        import weave
        from openai import OpenAI
        weave.init(f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}")
        client = OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"],
                        project=f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}")
        op = weave.op(name="crosswise_obstacles")(lambda seg_id, path: look_for_obstacles(path, client))
        with ThreadPoolExecutor(6) as ex:
            for r, res in zip(todo, ex.map(lambda r: _safe(op, r["segment_id"], video_path(r["segment_id"])), todo)):
                cache[r["segment_id"]] = res
        OBSTACLES.write_text(json.dumps(cache, indent=1))
    return cache


def _safe(op, seg_id, path):
    try:
        return op(seg_id, path)
    except Exception as e:
        return {"obstacles": [], "free": [0, 1], "passable": True, "model": VISION_MODEL, "error": type(e).__name__}


# ---------- moment policy ----------

def analyse_clip(segs, dets, obstacles):
    res, prev = {}, None
    cfg = CAMERAS.get(segs[0]["camera_id"], CAMERAS["default"])
    # quiet time counts from the start of observation: no gap decision before the robot has watched the street
    last_conflict, walk_run, history = segs[0]["start_sec"], 0, []
    sig, sig_t, blink_t = "none", -99.0, -99.0
    # YOLO misses small signal heads in many frames; the head does not move, so fall back to the nearest detection
    seen_heads = []
    for seg, det in zip(segs, dets):
        H, W = det["video_shape"][:2]
        for f in det["frames"][::STEP]:
            for d in f["detections"]:
                x1, y1, x2, y2 = d["bbox"]
                if (d["label"] == "traffic light" and cfg["head"] and cfg["head"][0] <= (x1 + x2) / 2 / W <= cfg["head"][1]
                        and y2 - y1 >= 45 and 0.7 <= (y2 - y1) / max(1, x2 - x1) <= 1.45):
                    seen_heads.append((seg["start_sec"] + f["time_sec"], [int(v) for v in d["bbox"]], W, H))
    for seg, det in zip(segs, dets):
        H, W = det["video_shape"][:2]
        frames = det["frames"][::STEP]
        imgs = decode(video_path(seg["segment_id"]), W, H)
        tracked = track(frames, W, H, STEP / (det.get("fps") or 30.0), prev)
        prev = tracked[-1] if tracked else prev
        ob = obstacles.get(seg["segment_id"]) or {"obstacles": [], "free": [0, 1], "passable": True}
        free_w = ob["free"][1] - ob["free"][0]
        tagged_block = bool(PATH_HAZARDS & set(seg.get("hazards") or [])) or (seg.get("cosmos_tags") or {}).get("blocked") == "yes"
        path_blocked = (not ob["passable"]) or free_w < 0.12
        px0, px1 = cfg["x0"], cfg["x1"]
        if ob["obstacles"] and not path_blocked:
            px0, px1 = max(px0, ob["free"][0]), min(px1, ob["free"][1])
            if px1 - px0 < 0.12:
                px0, px1 = cfg["x0"], cfg["x1"]
        out_frames, moments = [], []
        for k, (f, cur) in enumerate(zip(frames, tracked)):
            t = round(seg["start_sec"] + f["time_sec"], 2)
            # pedestrian signal
            head, reading = None, None
            confirm = WALK_CONFIRM
            if k < len(imgs) and cfg["head"]:
                heads = [l for l in cur["l"] if cfg["head"][0] <= (l["b"][0] + l["b"][2]) / 2 <= cfg["head"][1]
                         and read_head(imgs[k], l["px"]) is not None]
                if heads:
                    hb = max(heads, key=lambda l: l["px"][3] - l["px"][1])
                    reading = read_head(imgs[k], hb["px"])
                    head = r3(hb["b"]) + [reading]
                elif seen_heads:
                    _, px, hw, hh = min(seen_heads, key=lambda s: abs(s[0] - t))
                    reading, confirm = read_head(imgs[k], px), WALK_CONFIRM_REMEMBERED
                    if reading:
                        head = r3([px[0] / hw, px[1] / hh, px[2] / hw, px[3] / hh]) + [reading]
            # walk -> flashing hand (countdown) -> steady hand: a hand right after walk starts the countdown, which
            # lasts while the lamp keeps blinking off; lit for STEADY_S without a blink = steady don't-walk
            if reading == "walk":
                walk_run += 1
                if walk_run >= confirm:
                    sig, sig_t = "walk", t
            elif reading == "hand":
                walk_run = 0
                if sig == "walk":
                    sig, blink_t = "flashing", t
                elif sig != "flashing" or t - blink_t > STEADY_S:
                    sig = "dont_walk"
                sig_t = t
            elif reading == "dark" and sig == "flashing":
                walk_run, blink_t, sig_t = 0, t, t
            elif t - sig_t > SIGNAL_HOLD_S:
                sig = "none"
            # vehicles: now and predicted
            # vehicles in the robot's path now, or predicted to be within PREDICT_S; a vehicle already past the path
            # and driving away from it is no conflict
            moving = [o for o in cur["v"] if o["age"] >= 1 and speed(o) >= MOVING_V]
            in_path = lambda a, b: b >= px0 and a <= px1
            receding = lambda o: (o["b"][2] < px0 and o["vx"] <= 0) or (o["b"][0] > px1 and o["vx"] >= 0)
            now = [o for o in moving if o["b"][3] >= cfg["v"] and in_path(o["b"][0], o["b"][2])]
            soon = [o for o in moving if o not in now and not receding(o)
                    and o["b"][3] + max(0.0, o["vy"]) * PREDICT_S * W / H >= cfg["v"]
                    and in_path(o["b"][0] + o["vx"] * PREDICT_S, o["b"][2] + o["vx"] * PREDICT_S)]
            xp = [o for o in cur["p"] if o["age"] >= 1 and cfg["p0"] <= o["b"][3] <= cfg["p1"] and speed(o) >= MOVING_P]
            if now or soon:
                last_conflict = t
            history = (history + [bool(now)])[-50:]
            heavy = sum(history) / len(history) >= HEAVY_SHARE if len(history) >= 10 else False
            quiet = t - last_conflict
            need_quiet = QUIET_HEAVY if heavy else QUIET_LIGHT
            n = len(now)
            if now:
                d, why = "NO-GO", f"{n} vehicle{'s' if n > 1 else ''} moving through the crossing"
            elif soon:
                d, why = "NO-GO", f"vehicle approaching — will reach the crossing within {PREDICT_S:.1f} s"
            elif path_blocked:
                d, why = "WAIT", "crosswalk obstructed — no free path"
            elif sig == "dont_walk":
                d, why = "WAIT", "don't-walk signal" + (f" — {len(xp)} people jaywalking, robot waits" if xp else "")
            elif sig == "flashing":
                d, why = "WAIT", "countdown flashing — not enough time to cross"
            elif sig == "walk":
                d, why = "GO", "walk signal on, no vehicle in or approaching the crossing"
            elif quiet >= need_quiet:
                d, why = "GO", (f"gap: crossing quiet {quiet:.1f} s, nothing approaching" +
                                (" (heavy traffic, waited for a longer gap)" if heavy else ""))
            else:
                d, why = "WAIT", f"{'heavy traffic — ' if heavy else ''}confirming the gap ({quiet:.1f}/{need_quiet:.1f} s)"
            if d == "GO" and ob["obstacles"]:
                why += f"; steering through the free path ({ob['free'][0]:.2f}–{ob['free'][1]:.2f})"
            moments.append({"t": t, "d": d, "why": why, "sig": sig, "mv": n, "pred": len(soon), "xp": len(xp),
                            "lvl": "heavy" if heavy else "light", "quiet": round(min(quiet, 99), 1)})
            out_frames.append({"t": t,
                               "v": [r3(o["b"]) + [2 if (o in now or o in soon) else int(o in moving), o["c"],
                                                   round(o["vx"], 3), round(o["vy"], 3)] for o in cur["v"]],
                               "p": [r3(o["b"]) + [int(o in xp)] for o in cur["p"]],
                               "h": head})
        res[seg["segment_id"]] = {"cfg": {**cfg, "cross_s": round(cfg["width_m"] / ROBOT_MPS, 1)},
                                  "obstacles": {**ob, "path_blocked": path_blocked, "tagged_blocked": tagged_block,
                                                "path": [round(px0, 3), round(px1, 3)]},
                                  "frames": out_frames, "moments": moments}
    return res


def main(debug=False, refresh=False):
    rows = [r for r in json.loads((DATA / "decisions.json").read_text()) if r["caption_source"] != "vss_index"]
    with ThreadPoolExecutor(6) as ex:
        dets = dict(zip([r["segment_id"] for r in rows],
                        ex.map(lambda r: detections(r["segment_id"]), rows)))
        list(ex.map(lambda r: video_path(r["segment_id"]), rows))
    obstacles = obstacles_for(rows, refresh)
    clips = {}
    for r in rows:
        clips.setdefault(r["original_video"], []).append(r)
    res = {}
    for segs in clips.values():
        segs.sort(key=lambda x: x["segment_number"])
        res.update(analyse_clip(segs, [dets[s["segment_id"]] for s in segs], obstacles))
    OUT.write_text(json.dumps(res, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(res)} segments)")
    for v, segs in clips.items():
        print("\n" + v.rsplit("/", 1)[-1])
        for s in segs:
            ms = res[s["segment_id"]]["moments"]
            ob = res[s["segment_id"]]["obstacles"]
            strip = "".join({"GO": "G", "WAIT": ".", "NO-GO": "X"}[m["d"]] for m in ms)
            sigs = "".join({"walk": "w", "dont_walk": "h", "flashing": "f", "none": " "}[m["sig"]] for m in ms)
            print(f"  seg {s['segment_number']} LLM={s['decision']:5s} {strip} sig[{sigs}] obst={[o['type'] for o in ob['obstacles']]}"
                  f" free={ob['free']}" + (f" lvl={ms[-1]['lvl']}" if debug else ""))


if __name__ == "__main__":
    main(debug="--debug" in sys.argv, refresh="--refresh-obstacles" in sys.argv)
