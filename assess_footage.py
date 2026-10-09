"""Phase 1: score search hits with a caption rubric and rank parent videos. Reads tmp/phase1_search.json."""
import collections
import json
import pathlib
import re
import sys

TMP = pathlib.Path(__file__).resolve().parent.parent / "tmp"
CROSSING = r"crosswalk|zebra|crossing|intersection"
RUBRIC = {
    "pedestrians waiting at a crosswalk": [CROSSING, r"\bwait|standing (at|on|near)|curb|about to cross"],
    "pedestrians crossing while cars wait at a red light": [r"crossing|cross the (street|road)|crosswalk",
                                                            r"red (light|signal)|stopped|waiting|yield"],
    "vehicle turning through a crosswalk while people cross": [r"\bturn", r"crosswalk|pedestrian|crossing"],
    "pedestrian hesitates at the curb then crosses": [r"hesitat|paus|step(s|ping)? (off|onto)|look(s|ing)? both|before crossing"],
    "sidewalk or crosswalk blocked by scaffolding or construction": [r"scaffold|construction|traffic cone|\bcones?\b|barrier|barricade|blocked|closed|obstruct"],
}


def clear(q, cap):
    cap = (cap or "").lower()
    return all(re.search(p, cap) for p in RUBRIC[q])


if __name__ == "__main__":
    d = json.loads((TMP / "phase1_search.json").read_text())
    vid = collections.defaultdict(lambda: {"clear": 0, "queries": set(), "camera": None, "total": None, "srcs": set()})
    table = {}
    for q, cams in d.items():
        table[q] = {}
        for cam, v in cams.items():
            hits = v.get("hits", [])
            ok = [h for h in hits if clear(q, h["caption"])]
            table[q][cam] = len(ok)
            for h in ok:
                rec = vid[h["original_video"]]
                if h["source"] not in rec["srcs"]:
                    rec["srcs"].add(h["source"])
                    rec["clear"] += 1
                rec["queries"].add(q)
                rec["camera"], rec["total"] = cam, h["total_segments"]
    cams = list(next(iter(d.values())).keys())
    print("clear hits (of top 15) per query x camera")
    print(f"{'':58s}" + "".join(f"{c.replace('_streets_cam','').replace('nyc_bike_gopro','gopro'):>8s}" for c in cams) + "   TOTAL")
    for q in table:
        print(f"{q[:57]:58s}" + "".join(f"{table[q][c]:8d}" for c in cams) + f"{sum(table[q].values()):8d}")
    print("\nper camera total:", {c: sum(table[q][c] for q in table) for c in cams})
    print("\ntop parent videos by distinct clear segments")
    for v, r in sorted(vid.items(), key=lambda kv: (-len(kv[1]["queries"]), -kv[1]["clear"]))[:15]:
        print(f"  {r['clear']:2d} segs, {len(r['queries'])} queries, {r['total']} total segs  {r['camera']:18s} {v.rsplit('/', 1)[-1]}")
    if len(sys.argv) > 1:
        q = sys.argv[1]
        for cam, v in d[q].items():
            for h in v["hits"][:int(sys.argv[2]) if len(sys.argv) > 2 else 2]:
                print(f"\n[{cam}] {h['score']:.3f} clear={clear(q, h['caption'])} {h['source'].rsplit('/', 1)[-1]}\n  {h['caption'][:330]}")
