"""Caption crossing segments directly with NVIDIA Cosmos Reason using the CrossWise prompt.

Fallback for when DataEngine re-ingest is backlogged: same prompt, same model family,
called on the GPU host directly. Output: data/cosmos_captions.json (one row per segment).

Usage: python caption.py [--fill] [original_video ...]   (defaults to reingest.DEMO_VIDEOS;
       --fill re-captions only segments whose caption has no description)
"""
import base64
import json
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import vss
from reingest import CROSSWISE_PROMPT, DEMO_VIDEOS

COSMOS_URL = os.environ.get("COSMOS3_REASON_URL", "http://166.19.38.112:8001")
OUT = pathlib.Path(__file__).resolve().parent / "data" / "cosmos_captions.json"
TAG_RE = re.compile(r"\b(SIGNAL|VEHICLES|PEDS|BLOCKED|CROSS)\s*=\s*<?([a-z\-]+)>?", re.I)


def _gpu_headers():
    return {"Authorization": f"Bearer {os.environ['GPU_BEARER_TOKEN']}"}


def cosmos_model():
    r = requests.get(f"{COSMOS_URL}/v1/models", headers=_gpu_headers(), timeout=15)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def parse_tags(text):
    return {k.lower(): v.lower() for k, v in TAG_RE.findall(text or "")}


def has_description(text):
    return len(TAG_RE.sub("", text or "").strip()) >= 40


def caption_segment(seg, model):
    video = requests.get(f"{vss.BASE}/videos/stream", params={"source": seg["source"], "token": vss.login()},
                         timeout=120)
    video.raise_for_status()
    b64 = base64.b64encode(video.content).decode()
    t0 = time.time()
    text = ""
    for attempt in range(4):
        r = requests.post(f"{COSMOS_URL}/v1/chat/completions", headers=_gpu_headers(), timeout=240, json={
            "model": model, "max_tokens": 400, "temperature": 0.3 * attempt,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                {"type": "text", "text": CROSSWISE_PROMPT}]}]})
        if r.ok:
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if has_description(text):
                break
        time.sleep(2)
    r.raise_for_status()
    return {**seg, "caption": text, "tags": parse_tags(text), "caption_model": model,
            "caption_latency_s": round(time.time() - t0, 2)}


def segments_for(original_video):
    segs = vss.get("/tools/segments", original_video=original_video)["segments"]
    return [{"original_video": original_video, "source": s["source"], "camera_id": s.get("camera_id"),
             "location": s.get("location"), "segment_number": s.get("segment_number"),
             "start_sec": s.get("segment_start_sec"), "end_sec": s.get("segment_end_sec"),
             "object_counts": s.get("object_counts")}
            for s in sorted(segs, key=lambda s: s.get("segment_number") or 0)]


def main(videos, fill_only=False):
    model = cosmos_model()
    existing = {r["source"]: r for r in json.loads(OUT.read_text())} if OUT.exists() else {}
    segs = [s for v in videos for s in segments_for(v)]
    if fill_only:
        segs = [s for s in segs if not has_description(existing.get(s["source"], {}).get("caption"))]
    print(f"captioning {len(segs)} segments with {model}")
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(lambda s: caption_segment(s, model), segs))
    existing.update({r["source"]: r for r in rows})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(list(existing.values()), indent=1))
    for r in rows:
        print(f"{r['camera_id']:18s} #{r['segment_number']} {r['tags']}")
    print("wrote", OUT)


if __name__ == "__main__":
    fill = "--fill" in sys.argv
    main([a for a in sys.argv[1:] if a != "--fill"] or DEMO_VIDEOS, fill_only=fill)
