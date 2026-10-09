"""Replay the same delivery robot under two crossing policies on every demo clip.

  segment  one LLM verdict per 5 s segment (decide.py: Cosmos caption -> Llama 3.3 70B + guardrail)
  moment   frame-level perception every 0.2 s (perceive.py: YOLO tracks, pixel-read walk signal, Gemma 4 obstacles)

Robot: waits at the curb, starts on GO, crosses at 1.6 m/s (cfg.cross_s), next robot arrives 1.6 s after it is across.
Checked in hindsight against what perceive.py measured on the frames:
  unsafe_start   a vehicle moved through the path at the start or within the next 2 s (the robot has to yield; on
                 a fixed camera this includes cars that enter the field of view after the start)
  against_signal started while the pedestrian signal read don't-walk or the countdown was flashing
  missed_walk_s  how long after the walk signal lit the robot was still at the curb (first walk phase)
Output: data/policy_eval.json + a table on stdout.   Usage: python policy_eval.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
ARRIVE_S, CHECK_S = 1.6, 2.0


def simulate(moments, go, cross_s):
    starts, st, p_end, back = [], "curb", 0.0, 0.0
    for i, m in enumerate(moments):
        t = m["t"]
        if st == "crossing" and t >= p_end:
            st, back = "across", t + ARRIVE_S
        if st == "across" and t >= back:
            st = "curb"
        if st == "curb" and go(i, m):
            starts.append(i)
            st, p_end = "crossing", t + cross_s
    return starts


def check(moments, starts):
    unsafe = against = 0
    for i in starts:
        t = moments[i]["t"]
        if any(m["mv"] for m in moments[i:] if m["t"] <= t + CHECK_S):
            unsafe += 1
        if moments[i]["sig"] in ("dont_walk", "flashing"):
            against += 1
    return unsafe, against


def main():
    P = json.loads((DATA / "perception.json").read_text())
    rows = [r for r in json.loads((DATA / "decisions.json").read_text()) if r["segment_id"] in P]
    clips = {}
    for r in rows:
        clips.setdefault(r["original_video"], []).append(r)
    out = []
    for v, segs in clips.items():
        segs.sort(key=lambda r: r["segment_number"])
        moments = [m for s in segs for m in P[s["segment_id"]]["moments"]]
        frames = [f for s in segs for f in P[s["segment_id"]]["frames"]]
        cross_s = P[segs[0]["segment_id"]]["cfg"]["cross_s"]
        seg_of = lambda t: next((s for s in segs if s["start_sec"] <= t < s["end_sec"]), segs[-1])
        # walk onset = first frame the walking-person lamp is lit after a don't-walk phase (before confirmation)
        walk_on = next((f["t"] for f, m in zip(frames, moments) if f["h"] and f["h"][4] == "walk"
                        and m["sig"] in ("dont_walk", "flashing")
                        and any(x["sig"] == "walk" for x in moments if f["t"] < x["t"] <= f["t"] + 1.5)), None)
        res = {"video": v, "camera_id": segs[0]["camera_id"], "cross_s": cross_s, "duration_s": moments[-1]["t"] + 0.2,
               "signal_read": any(m["sig"] != "none" for m in moments), "walk_onset_s": walk_on}
        for name, go in (("segment", lambda i, m: seg_of(m["t"])["decision"] == "GO"),
                         ("moment", lambda i, m: m["d"] == "GO")):
            starts = simulate(moments, go, cross_s)
            unsafe, against = check(moments, starts)
            first = moments[starts[0]]["t"] if starts else None
            after = [moments[i]["t"] for i in starts if walk_on is not None and moments[i]["t"] >= walk_on]
            res[name] = {"crossings": len(starts), "first_start_s": first, "unsafe_start": unsafe,
                         "against_signal": against, "start_times": [moments[i]["t"] for i in starts],
                         "missed_walk_s": round(after[0] - walk_on, 1) if after else None}
        out.append(res)
    (DATA / "policy_eval.json").write_text(json.dumps(out, indent=1))
    tot = {k: {"crossings": 0, "unsafe_start": 0, "against_signal": 0} for k in ("segment", "moment")}
    print(f"{'clip':34s} {'policy':8s} cross  first  unsafe  vs-signal  walk->start")
    for r in out:
        name = r["camera_id"] + " " + r["video"].rsplit("_chunk_", 1)[-1][:4]
        for k in ("segment", "moment"):
            x = r[k]
            for f in tot[k]:
                tot[k][f] += x[f]
            print(f"{name:34s} {k:8s} {x['crossings']:5d}  {x['first_start_s'] if x['first_start_s'] is not None else '-':>5}"
                  f"  {x['unsafe_start']:6d}  {x['against_signal']:9d}  "
                  f"{(str(x['missed_walk_s']) + ' s') if x['missed_walk_s'] is not None else ('never' if r['walk_onset_s'] is not None else '-'):>10}")
    print("\ntotal", json.dumps(tot))


if __name__ == "__main__":
    main()
