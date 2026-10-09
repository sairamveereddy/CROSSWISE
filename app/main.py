"""CrossWise web app backend (stdlib only, runs on python:3.12-slim with code from a ConfigMap).

Routes (Ingress strips /app, so everything is served from /):
  GET /                      single-page UI
  GET /health                liveness/readiness
  GET /api/clips             clips with decisions (demo clips first)
  GET /api/timeline?video=&n=  per-segment decisions + escalations for one clip
  GET /api/scorecard?n=      crossing difficulty ranking per camera
  GET /api/search?q=         VSS semantic search, restricted to the street cameras
  GET /video?source=         VSS video proxy with Range support (the JWT never reaches the browser)
  GET /label                 blind human-labelling page (20 segments, safe / unsafe)
  GET|POST /api/labels       read / save labels; GET /api/label-set lists the segments to label

Decision support on recorded footage only. It does not control a robot.
"""
import json
import os
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if (p / "agent.py").exists():
        sys.path.insert(0, str(p))
import agent  # noqa: E402


def _find(name):
    for p in (HERE / name, HERE.parent / "data" / name):
        if p.exists():
            return p
    raise FileNotFoundError(name)


VSS_URL = os.environ["VSS_URL"].rstrip("/")
VSS_USER = os.environ["VSS_USERNAME"]
VSS_PASS = os.environ["VSS_PASSWORD"]
PORT = int(os.environ.get("PORT", "8080"))
ALLOWED_SOURCE = re.compile(r"^s3://team-48-vss-chunks(-segments)?/[\w./-]+\.mp4$")
TAG_RE = re.compile(r"\s*\b(SIGNAL|VEHICLES|PEDS|BLOCKED|CROSS)\s*=\s*<?[a-z\-]+>?", re.I)

DECISIONS = json.loads(_find("decisions.json").read_text())
CAPTIONS = {r["source"]: TAG_RE.sub("", r["caption"]).strip()
            for r in json.loads(_find("cosmos_captions.json").read_text())}
INDEX_HTML = (HERE / "index.html").read_bytes()
LABEL_HTML = (HERE / "label.html").read_bytes()
LOGO = (HERE / "logo.jpg").read_bytes() if (HERE / "logo.jpg").exists() else b""
try:
    LABEL_SET = json.loads(_find("label_set.json").read_text())
except FileNotFoundError:
    LABEL_SET = []
LABELS_FILE = pathlib.Path(os.environ.get("LABELS_FILE", "/tmp/crosswise_labels.json"))
_labels_lock = threading.Lock()


def read_labels():
    return json.loads(LABELS_FILE.read_text()) if LABELS_FILE.exists() else {}


class VSS:
    _token, _at, _lock = None, 0.0, threading.Lock()

    @classmethod
    def token(cls, force=False):
        with cls._lock:
            if force or not cls._token or time.time() - cls._at > 1500:
                body = json.dumps({"username": VSS_USER, "password": VSS_PASS}).encode()
                req = urllib.request.Request(f"{VSS_URL}/api/v1/auth/login", body,
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    cls._token, cls._at = json.load(r)["access_token"], time.time()
            return cls._token

    @classmethod
    def request(cls, path, body=None, headers=None, timeout=120):
        for attempt in range(2):
            h = {"Authorization": f"Bearer {cls.token(force=attempt > 0)}", **(headers or {})}
            data = None
            if body is not None:
                data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
            try:
                return urllib.request.urlopen(urllib.request.Request(f"{VSS_URL}/api/v1{path}", data, h),
                                              timeout=timeout)
            except urllib.error.HTTPError as e:
                if e.code != 401 or attempt:
                    raise


def clip_list():
    out = []
    for v, segs in agent.timelines(DECISIONS).items():
        ds = [s["decision"] for s in segs]
        out.append({"video": v, "name": v.rsplit("/", 1)[-1], "camera_id": segs[0]["camera_id"],
                    "location": segs[0].get("location"), "demo": segs[0]["caption_source"] != "vss_index",
                    "segments": len(segs), "go": ds.count("GO"), "wait": ds.count("WAIT"), "nogo": ds.count("NO-GO")})
    out.sort(key=lambda c: (not c["demo"], c["camera_id"], c["name"]))
    return out


def timeline(video, n):
    tls = agent.timelines(DECISIONS)
    if video not in tls:
        return None
    segs = tls[video]
    card = agent.scorecard(DECISIONS, n)
    alt = agent.easiest_alternative(card, segs[0]["camera_id"], segs[0].get("location"))
    out = []
    for s in segs:
        out.append({k: s.get(k) for k in ("segment_id", "segment_number", "start_sec", "end_sec", "decision",
                                          "llm_decision", "unclear", "signal", "hazards", "reasons", "confidence",
                                          "yolo", "cosmos_tags", "caption_source", "model")}
                   | {"caption": CAPTIONS.get(s["segment_id"])})
    return {"video": video, "camera_id": segs[0]["camera_id"], "location": segs[0].get("location"), "n": n,
            "segments": out, "escalations": agent.detect_stuck(segs, n, alt)}


def search(q):
    r = json.load(VSS.request("/search", {"query": q, "top_k": 40, "llm_top_n": 1, "min_similarity": 0.2}))
    known = {d["segment_id"]: d for d in DECISIONS}
    cams = set(agent_cameras())
    hits = []
    for h in r.get("results", []):
        if h.get("camera_id") not in cams:
            continue
        d = known.get(h["source"])
        hits.append({"source": h["source"], "video": h.get("original_video"), "camera_id": h.get("camera_id"),
                     "start_sec": h.get("segment_start_sec"), "end_sec": h.get("segment_end_sec"),
                     "segment_number": h.get("segment_number"), "score": round(h.get("similarity_score") or 0, 3),
                     "decision": d["decision"] if d else None, "reasons": d["reasons"] if d else None,
                     "objects": h.get("object_counts")})
    return hits[:15]


def agent_cameras():
    return sorted({d["camera_id"] for d in DECISIONS})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if "/health" not in self.path:
            sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))

    def _send(self, code, body, ctype="application/json", extra=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        n = max(2, min(12, int(qs.get("n", agent.DEFAULT_N))))
        try:
            if url.path in ("/", "/index.html"):
                return self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            if url.path == "/logo.jpg":
                return self._send(200, LOGO, "image/jpeg", {"Cache-Control": "public, max-age=86400"})
            if url.path in ("/label", "/label/", "/label.html"):
                return self._send(200, LABEL_HTML, "text/html; charset=utf-8")
            if url.path == "/api/label-set":
                return self._send(200, LABEL_SET)
            if url.path == "/api/labels":
                return self._send(200, read_labels())
            if url.path == "/health":
                return self._send(200, {"ok": True, "segments": len(DECISIONS)})
            if url.path == "/api/clips":
                return self._send(200, clip_list())
            if url.path == "/api/timeline":
                t = timeline(qs.get("video", ""), n)
                return self._send(200, t) if t else self._send(404, {"error": "unknown clip"})
            if url.path == "/api/scorecard":
                return self._send(200, agent.scorecard(DECISIONS, n))
            if url.path == "/api/search":
                q = qs.get("q", "").strip()[:200]
                return self._send(200, search(q) if q else [])
            if url.path == "/video":
                return self._video(qs.get("source", ""))
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            sys.stderr.write(f"error {url.path}: {type(e).__name__}: {e}\n")
            try:
                self._send(502, {"error": type(e).__name__})
            except Exception:
                pass

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/labels":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 65536)) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        allowed = {i["segment_id"] for i in LABEL_SET}
        updates = {k: v for k, v in body.items() if k in allowed and v in ("safe", "unsafe", None)}
        with _labels_lock:
            labels = read_labels()
            for k, v in updates.items():
                if v is None:
                    labels.pop(k, None)
                else:
                    labels[k] = v
            LABELS_FILE.write_text(json.dumps(labels, indent=1))
        return self._send(200, {"saved": len(updates), "total": len(labels)})

    def _video(self, source):
        if not ALLOWED_SOURCE.match(source):
            return self._send(400, {"error": "source not allowed"})
        headers = {"Range": self.headers["Range"]} if self.headers.get("Range") else {}
        up = None
        for attempt in range(2):
            stream_url = f"{VSS_URL}/api/v1/videos/stream?" + urllib.parse.urlencode(
                {"source": source, "token": VSS.token(force=attempt > 0)})
            try:
                up = urllib.request.urlopen(urllib.request.Request(stream_url, headers=headers), timeout=120)
                break
            except urllib.error.HTTPError as e:
                if e.code != 401 or attempt:
                    return self._send(e.code, {"error": "upstream"})
        with up:
            self.send_response(up.status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            for k in ("Content-Length", "Content-Range"):
                if up.headers.get(k):
                    self.send_header(k, up.headers[k])
            self.send_header("Cache-Control", "private, max-age=600")
            self.end_headers()
            if self.command == "HEAD":
                return
            while chunk := up.read(256 * 1024):
                self.wfile.write(chunk)


if __name__ == "__main__":
    print(f"CrossWise listening on 0.0.0.0:{PORT} with {len(DECISIONS)} decisions", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
