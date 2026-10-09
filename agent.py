"""CrossWise agent behaviour on top of per-segment decisions (stdlib only; imported by the web app).

- Stuck detection: N consecutive non-GO segments (WAIT or NO-GO) fire one escalation per streak,
  with a reason and a recommended action (reroute or remote-operator handoff).
- Crossing scorecard per camera: share of GO time, mean wait before a GO, turning-vehicle hazards,
  NO-GO share and escalations, ranked easiest -> hardest.

Usage: python agent.py [--n 4] [--decisions data/decisions.json]  -> data/agent_output.json (+ Weave trace)
"""
import argparse
import collections
import json
import pathlib

DATA = pathlib.Path(__file__).resolve().parent / "data"
DEFAULT_N = 4
PATH_HAZARDS = {"blocked_crosswalk", "construction"}
TRAFFIC_HAZARDS = {"turning_vehicle", "vehicle_in_crosswalk", "moving_traffic", "emergency_vehicle"}


def dur(r):
    return max(0.0, (r.get("end_sec") or 0) - (r.get("start_sec") or 0)) or 5.0


def timelines(rows):
    """Group decision rows into per-parent-video timelines ordered by segment number (latest run wins)."""
    latest = {}
    for r in rows:
        k = r["segment_id"]
        if k not in latest or r.get("created_at", "") >= latest[k].get("created_at", ""):
            latest[k] = r
    out = collections.defaultdict(list)
    for r in latest.values():
        out[r["original_video"]].append(r)
    return {v: sorted(segs, key=lambda r: r["segment_number"]) for v, segs in out.items()}


def is_turning(r):
    return "turning_vehicle" in (r.get("hazards") or []) or (r.get("cosmos_tags") or {}).get("vehicles") == "turning"


def recommend(streak, alternative=None):
    hz = collections.Counter(h for r in streak for h in r.get("hazards") or [])
    n = len(streak)
    waited = sum(dur(r) for r in streak)
    path_blocked = sum(1 for r in streak if PATH_HAZARDS & set(r.get("hazards") or []))
    unclear = sum(1 for r in streak if r.get("unclear"))
    traffic = sum(1 for r in streak if TRAFFIC_HAZARDS & set(r.get("hazards") or []))
    signals = collections.Counter(r.get("signal") for r in streak)
    alt = f" Suggested alternative: {alternative}." if alternative else ""
    if path_blocked * 2 >= n:
        return ("REROUTE", f"Crosswalk obstructed (blocked/construction) in {path_blocked} of {n} segments "
                           f"over {waited:.0f}s; this crossing is not passable right now.{alt}")
    if unclear * 2 >= n:
        return ("REMOTE_OPERATOR", f"Evidence unclear in {unclear} of {n} segments over {waited:.0f}s; "
                                   "hand off to a remote operator to judge the crossing.")
    if traffic * 2 >= n:
        top = ", ".join(h.replace("_", " ") for h, _ in hz.most_common(2))
        return ("REMOTE_OPERATOR", f"No safe gap for {waited:.0f}s ({top}); request remote-operator "
                                   f"assistance or reroute.{alt}")
    sig = signals.most_common(1)[0][0] if signals else "unclear"
    return ("REMOTE_OPERATOR", f"Waiting {waited:.0f}s without a GO (signal mostly {sig}); "
                               "hand off to a remote operator.")


def detect_stuck(timeline, n=DEFAULT_N, alternative=None):
    """Return escalations: one per maximal non-GO streak, fired at the segment where the streak reaches n."""
    escalations, streak = [], []
    for r in timeline:
        if r["decision"] == "GO":
            streak = []
            continue
        streak.append(r)
        if len(streak) == n:
            action, reason = recommend(streak, alternative)
            escalations.append({"segment_id": r["segment_id"], "segment_number": r["segment_number"],
                                "at_sec": r.get("end_sec"), "streak_start_sec": streak[0].get("start_sec"),
                                "waited_s": sum(dur(x) for x in streak), "action": action, "reason": reason})
    return escalations


def waits_before_go(timeline):
    """Lengths (s) of non-GO streaks: (completed streaks that end in a GO, censored streaks that never reach one)."""
    done, censored, cur = [], [], 0.0
    for r in timeline:
        if r["decision"] == "GO":
            done.append(cur)
            cur = 0.0
        else:
            cur += dur(r)
    if cur:
        censored.append(cur)
    return done, censored


def scorecard(rows, n=DEFAULT_N):
    tls = timelines(rows)
    per = collections.defaultdict(lambda: {"segments": 0, "seconds": 0.0, "go_s": 0.0, "nogo_s": 0.0, "unclear": 0,
                                           "turning": 0, "escalations": 0, "waits": [], "censored": [], "clips": 0,
                                           "location": ""})
    for tl in tls.values():
        c = per[tl[0]["camera_id"]]
        c["clips"] += 1
        c["location"] = tl[0].get("location") or c["location"]
        for r in tl:
            c["segments"] += 1
            c["seconds"] += dur(r)
            c["go_s"] += dur(r) if r["decision"] == "GO" else 0
            c["nogo_s"] += dur(r) if r["decision"] == "NO-GO" else 0
            c["unclear"] += bool(r.get("unclear"))
            c["turning"] += is_turning(r)
        c["escalations"] += len(detect_stuck(tl, n))
        w, cz = waits_before_go(tl)
        c["waits"] += w
        c["censored"] += cz
    out = []
    for cam, c in per.items():
        go_share = c["go_s"] / c["seconds"]
        nogo_share = c["nogo_s"] / c["seconds"]
        # censored streaks count at their observed length, so this is a lower bound on the true wait
        all_waits = c["waits"] + c["censored"]
        avg_wait = sum(all_waits) / len(all_waits) if all_waits else 0.0
        turning_rate = c["turning"] / c["segments"]
        difficulty = round(100 * (0.5 * (1 - go_share) + 0.2 * nogo_share + 0.2 * min(avg_wait / 30, 1)
                                  + 0.1 * min(turning_rate * 4, 1)), 1)
        out.append({"camera_id": cam, "location": c["location"], "clips": c["clips"], "segments": c["segments"],
                    "go_share": round(go_share, 3), "nogo_share": round(nogo_share, 3),
                    "avg_wait_before_go_s": round(avg_wait, 1), "gos_reached": len(c["waits"]),
                    "streaks_without_go": len(c["censored"]), "turning_hazards": c["turning"],
                    "unclear_segments": c["unclear"], "escalations": c["escalations"], "difficulty": difficulty})
    out.sort(key=lambda s: s["difficulty"])
    for i, s in enumerate(out, 1):
        s["rank"] = i
    return out


def easiest_alternative(card, camera_id, location):
    for s in card:
        if s["camera_id"] != camera_id and s["location"] == location:
            return f"{s['camera_id']} (difficulty {s['difficulty']}, GO {s['go_share']:.0%} of the time)"
    return None


def build(rows, n=DEFAULT_N):
    card = scorecard(rows, n)
    videos = {}
    for v, tl in timelines(rows).items():
        alt = easiest_alternative(card, tl[0]["camera_id"], tl[0].get("location"))
        videos[v] = {"camera_id": tl[0]["camera_id"], "location": tl[0].get("location"),
                     "caption_source": tl[0]["caption_source"], "segments": tl,
                     "escalations": detect_stuck(tl, n, alt)}
    return {"n": n, "scorecard": card, "videos": videos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--decisions", default=str(DATA / "decisions.json"))
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()
    rows = json.loads(pathlib.Path(args.decisions).read_text())
    run = build
    if not args.no_trace:
        import os
        import weave
        weave.init(f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}")
        run = weave.op(name="crosswise_agent_build")(build)
    out = run(rows, args.n)
    (DATA / "agent_output.json").write_text(json.dumps(out, indent=1))
    print(f"N={args.n}  escalations:")
    for v, d in out["videos"].items():
        for e in d["escalations"]:
            if d["caption_source"] != "vss_index":
                print(f"  {d['camera_id']:18s} {v.rsplit('/', 1)[-1][:40]:40s} @{e['at_sec']:.0f}s {e['action']}: {e['reason']}")
    print(f"  (+{sum(len(d['escalations']) for d in out['videos'].values() if d['caption_source'] == 'vss_index')} in scorecard sample clips)")
    print("\nscorecard (easiest -> hardest):")
    for s in out["scorecard"]:
        print(f"  {s['rank']}. {s['camera_id']:18s} diff={s['difficulty']:5.1f} GO={s['go_share']:5.0%} NO-GO={s['nogo_share']:4.0%} "
              f"wait~{s['avg_wait_before_go_s']:4.1f}s turning={s['turning_hazards']:2d} esc={s['escalations']} clips={s['clips']}")


if __name__ == "__main__":
    main()
