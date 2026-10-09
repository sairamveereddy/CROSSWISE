"""Phase 2: re-ingest the chosen crossing chunks with the CrossWise caption prompt.

One whole chunk per job (chunk_count=1, addressed by original_video, never stream_id).
Writes job ids to tmp/reingest_jobs.json and polls every 4 s until all jobs finish.
"""
import json
import pathlib
import sys
import time

import vss

TMP = pathlib.Path(__file__).resolve().parent.parent / "tmp"

CROSSWISE_PROMPT = (
    "View of a street crossing. Describe for a sidewalk robot deciding whether to cross: "
    "traffic light and walk signal state if visible, vehicles moving, stopped, or turning into "
    "the crosswalk, pedestrians waiting or crossing, cyclists, and anything blocking the curb or "
    "crosswalk. Do not describe individual people. End with one line: "
    "SIGNAL=<green|red|yellow|none|unclear> VEHICLES=<stopped|moving|turning|none> "
    "PEDS=<crossing|waiting|none> BLOCKED=<yes|no> CROSS=<go|wait|no-go|unclear>"
)

DEMO_VIDEOS = [
    "s3://team-48-vss-chunks/team-48/20261001_101322_sf4_chunk_0015.mp4",
    "s3://team-48-vss-chunks/team-48/20261001_094630_sf1_chunk_0013.mp4",
    "s3://team-48-vss-chunks/team-48/20261008_065653_VID_20261006_171812_002_chunk_0029.mp4",
]


def start(videos):
    assert len(CROSSWISE_PROMPT) <= 800, len(CROSSWISE_PROMPT)
    jobs = {}
    for v in videos:
        r = vss.post("/dashboard/reingest", {"original_video": v, "chunk_count": 1,
                                             "custom_prompt": CROSSWISE_PROMPT})
        jobs[v] = r.get("job_id")
        print("started", v.rsplit("/", 1)[-1], {k: r.get(k) for k in ("job_id", "status", "copied_segments")})
    (TMP / "reingest_jobs.json").write_text(json.dumps(jobs, indent=1))
    return jobs


def monitor(jobs, timeout_s=1800):
    done = {}
    t0 = time.time()
    while len(done) < len(jobs) and time.time() - t0 < timeout_s:
        for v, jid in jobs.items():
            if v in done:
                continue
            try:
                s = vss.get(f"/dashboard/reingest/{jid}")
            except Exception as e:
                print(f"  {jid}: poll error {e}")
                continue
            status = s.get("status")
            print(f"  [{int(time.time() - t0):4d}s] {v.rsplit('/', 1)[-1][:40]:40s} {status}: "
                  f"{s.get('completed_chunks')}/{s.get('total_chunks')} chunks, "
                  f"{s.get('indexed_segments')}/{s.get('total_segments')} clips")
            if status in ("completed", "failed", "error"):
                done[v] = s
        sys.stdout.flush()
        time.sleep(4)
    return done


if __name__ == "__main__":
    print("prompt chars:", len(CROSSWISE_PROMPT))
    if sys.argv[1:] == ["monitor"]:
        jobs = json.loads((TMP / "reingest_jobs.json").read_text())
    else:
        jobs = start(DEMO_VIDEOS)
    result = monitor(jobs)
    print(json.dumps({v.rsplit("/", 1)[-1]: s.get("status") for v, s in result.items()}, indent=1))
