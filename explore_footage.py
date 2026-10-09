"""Phase 1: probe the corpus for street-crossing footage. Writes tmp/phase1_search.json."""
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

import vss

CAMERAS = ["nyc_streets_cam-1", "nyc_streets_cam-2", "sf_streets_cam-1", "sf_streets_cam-2",
           "sf_streets_cam-3", "sf_streets_cam-4", "sf_streets_cam-5", "pie_cam-3", "nyc_bike_gopro-1"]
QUERIES = [
    "pedestrians waiting at a crosswalk",
    "pedestrians crossing while cars wait at a red light",
    "vehicle turning through a crosswalk while people cross",
    "pedestrian hesitates at the curb then crosses",
    "sidewalk or crosswalk blocked by scaffolding or construction",
]


def run(q, cam):
    try:
        r = vss.post("/search", {"query": q, "top_k": 15, "min_similarity": 0.3, "llm_top_n": 1,
                                 "metadata_filters": {"camera_id": cam}})
    except Exception as e:  # keep going; report per-cell errors
        return q, cam, {"error": str(e)}
    hits = [{"source": h.get("source"), "score": h.get("similarity_score"),
             "caption": h.get("reasoning_content"), "original_video": h.get("original_video"),
             "start": h.get("segment_start_sec"), "end": h.get("segment_end_sec"),
             "segment_number": h.get("segment_number"), "total_segments": h.get("total_segments"),
             "object_counts": h.get("object_counts"), "camera_id": cam}
            for h in r.get("results", [])]
    chunks = [{"original_video": c.get("original_video"), "matched": c.get("matched_segment_count"),
               "best_start": c.get("best_match_start_sec"), "best_end": c.get("best_match_end_sec")}
              for c in r.get("chunk_results", [])]
    return q, cam, {"hits": hits, "chunks": chunks, "result_keys": sorted(r.get("results", [{}])[0].keys()) if r.get("results") else []}


if __name__ == "__main__":
    vss.login()
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda qc: run(*qc), [(q, c) for q in QUERIES for c in CAMERAS]))
    data = {q: {} for q in QUERIES}
    for q, c, res in out:
        data[q][c] = res
    path = pathlib.Path(__file__).resolve().parent.parent / "tmp" / "phase1_search.json"
    path.write_text(json.dumps(data, indent=1))
    for q in QUERIES:
        print(q, {c: (len(v["hits"]) if "hits" in v else v["error"][:60]) for c, v in data[q].items()})
