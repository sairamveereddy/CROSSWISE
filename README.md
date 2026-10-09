# CrossWise: a crossing copilot for sidewalk delivery robots

CrossWise watches a street crossing and, for every 5-second moment, decides **GO**, **WAIT**, or **NO-GO**, with
reasons and video evidence. If the robot would be stuck waiting too long, it **escalates** (reroute or remote-operator
handoff) so the robot never sits at a curb indefinitely. It also ranks crossings by difficulty for route planning.

> **Decision support only.** CrossWise runs on *recorded* street footage. It does not control a robot, and its
> output must not be used as the sole input for any real-world crossing decision.

Built for the VAST Builders Challenge (team-48) on a pre-deployed video understanding and search stack.

## Sponsor stack

| Tool | Used for |
|---|---|
| **VAST DataEngine** | Ingest pipeline (segmenter → YOLO detector → Cosmos reasoner → embedder → writer) that indexed the archive; re-ingest API |
| **VastDB** | `vss-collection` (indexed segments, captions, YOLO counts, embeddings) and our own table `hackathon.crossing_decisions` |
| **NVIDIA Cosmos Reason** (on CoreWeave GPUs) | Crossing-specific captions with structured tags (`SIGNAL= VEHICLES= PEDS= BLOCKED= CROSS=`) |
| **NVIDIA Cosmos Embed + YOLO11** (on CoreWeave GPUs) | Semantic search over the archive; per-segment object counts used as evidence |
| **Weights & Biases Inference** (on CoreWeave) | `meta-llama/Llama-3.3-70B-Instruct` turns caption + detections into a JSON decision; `google/gemma-4-31B-it` (vision) finds obstacles and the free path on the crosswalk |
| **W&B Weave** | Traces every decision call and the agent build; W&B run logs the human evaluation |
| **Cursor** | Built end to end with the Cursor agent and the repo's VSS skills |

## How it works

```mermaid
flowchart LR
  A[Street footage in VAST S3] --> B[DataEngine: YOLO11 + Cosmos Reason + Embed]
  B --> C[(VastDB vss-collection)]
  C -->|search / segments API| D[caption.py: Cosmos Reason with crossing prompt]
  C -->|YOLO counts| E
  D --> E[decide.py: W&B Inference LLM + safety guardrail, traced in Weave]
  E --> F[(VastDB hackathon.crossing_decisions)]
  E --> G[agent.py: stuck detection + crossing scorecard]
  G --> H[Web app on Kubernetes /app]
  C -->|video stream proxy| H
```

1. **Find footage** (`explore_footage.py`, `assess_footage.py`): 5 crossing queries × 9 cameras through `POST /search`;
   captions scored with a keyword rubric to count clear hits. Fixed curb-level cameras (`sf_streets_cam-*`,
   `nyc_streets_cam-*`) were kept; the dashcam (`pie_cam-3`) and night bike GoPro were dropped (not a robot's viewpoint).
2. **Crossing captions** (`reingest.py`, `caption.py`): the 3 demo clips were submitted to DataEngine re-ingest with our
   crossing prompt. The shared pipeline was replaying a backlog and did not index them in time, so `caption.py` calls
   **Cosmos Reason directly** with the same prompt on each 5-second segment (≈4–6 s per call; 18 segments in under 30 s with 4 workers).
3. **Decide** (`decide.py`): caption + YOLO counts (+ Cosmos tags when present) → W&B Inference (JSON mode, temperature 0)
   → `{decision, unclear, signal, hazards[], reasons[], confidence}`.
   - GO only on clear evidence (walk/green, no vehicle in or turning through the crosswalk, path passable).
   - WAIT when it should clear soon, and **whenever evidence is unclear** (marked `unclear`).
   - NO-GO for immediate danger or no passable path.
   - **Guardrail in code:** a GO is downgraded to WAIT if Cosmos tagged `VEHICLES=turning` or `BLOCKED=yes`, or the model
     marked the evidence unclear. The model's raw answer is kept as `llm_decision` for evaluation.
   - Prompts forbid describing individuals or licence plates; index captions (which can describe people) are never shown.
4. **Agent** (`agent.py`, standard library only, imported by the app):
   - **Stuck detection:** N consecutive non-GO segments (default 4 = 20 s, configurable in the UI) fire one escalation
     per streak. Action = **REROUTE** if ≥ half the streak shows a blocked crosswalk/construction (with the easiest
     same-city crossing suggested), otherwise **REMOTE_OPERATOR** (unclear evidence, no safe traffic gap, or long signal wait).
   - **Crossing scorecard per camera:** share of GO time, mean wait before a GO, turning-vehicle hazards, NO-GO share,
     escalations. Difficulty 0–100 = 50% time without GO + 20% NO-GO share + 20% mean wait (capped 30 s) + 10% turning rate.
5. **Frame-level perception** (`perceive.py`): one verdict per 5 s segment is too coarse. A robot loses whole walk phases
   waiting for the next segment, and an LLM reading a caption confuses the vehicle light with the pedestrian signal. For the
   demo clips CrossWise decides every 0.2 s from what is measured on the frames:
   - **Vehicle tracks** from the YOLO11 boxes VAST stores for every frame, sampled at 5 fps. Motion uses both box
     edges: a person walking in front of a parked car, or the frame border, moves only one edge and is not motion. The
     camera's own shake (handheld NYC footage) is removed using the static signal heads. Speed is averaged over 0.6 s,
     and a vehicle's position is **predicted 1.5 s ahead**.
   - **Pedestrian signal read from the pixels:** the signal head facing the robot's crosswalk (configured per camera;
     other heads govern other crosswalks) is cropped and classified as lit hand, lit walking person or dark. A state
     machine turns that into **WALK**, **COUNTDOWN** (hand blinking after walk) or **DON'T WALK**. WALK needs 0.6 s of
     consistent readings, or 1.2 s when YOLO missed the head and its last known position is read.
   - **Obstacles on the crosswalk:** one frame per segment goes to **Gemma 4 (vision) on W&B Inference (CoreWeave)**,
     which returns cones, barriers or stopped vehicles and the widest free span. The robot steers through that span, and
     vehicles are checked against it.
   - **Policy:** **NO-GO** if a vehicle moves through the robot's path now or is predicted to within 1.5 s (vehicles past
     the path and driving away are ignored). **WAIT** on DON'T WALK, on COUNTDOWN (too late to start) or when the path is
     obstructed. **GO** the moment WALK is confirmed. Where no signal is visible, GO on a gap: the path is quiet for
     0.6 s in light traffic, or 2 s when vehicles were in the path ≥ 30% of the last 10 s.
   - **Robot:** curb → crossing at 1.6 m/s (crossing length per camera) → yield if a vehicle enters → across. A DON'T
     WALK it can read explains the wait, so it holds calmly instead of spinning and escalates only past 90 s (longer
     than a signal cycle). Without a signal it escalates after N×5 s.
   We also asked Cosmos Reason for a per-second signal state; it answered "walk" for every second of every segment, so
   the signal is read from the pixels instead.
   `policy_eval.py` replays the same robot under both policies (see Evaluation).
6. **App** (`app/`): stdlib Python backend + one HTML page (Chart.js from CDN), deployed with `python:3.12-slim`, code from
   a ConfigMap, VSS credentials from a Secret, Ingress `/app`. Video is proxied with HTTP Range support; the JWT never
   reaches the browser. Features: player with live GO/WAIT/NO-GO badge, reasons and evidence, decision timeline with a
   tiny robot riding the playhead, escalation banner, **replay mode** (steps through a clip as if live, labelled REPLAY),
   crossing scorecard, archive search, and a blind labelling page (`/app/label`).

## Repository layout

```
crosswise/
  vss.py               VSS retrieval API client (login, cached JWT, retry on 401)
  explore_footage.py   Phase 1 searches -> tmp/phase1_search.json
  assess_footage.py    Phase 1 rubric + parent-video ranking
  reingest.py          crossing prompt + DataEngine re-ingest (start / monitor)
  caption.py           direct Cosmos Reason captioning -> data/cosmos_captions.json
  decide.py            decision engine -> data/decisions.json + VastDB + Weave
  agent.py             stuck detection + scorecard -> data/agent_output.json
  perceive.py          5 fps tracks, pixel-read walk signal, Gemma 4 obstacles -> moment GO/WAIT/NO-GO -> data/perception.json
  policy_eval.py       same robot under segment-level vs frame-level policy -> data/policy_eval.json
  evaluate.py          label-set selection + human-eval scoring + W&B run
  app/main.py          web backend (stdlib)      app/index.html  UI      app/label.html  labelling UI
  deploy/k8s.yaml      Deployment + Service + Ingress (no secrets)      deploy/deploy.sh
  data/                generated outputs used by the app (no captions that describe people)
```

## Setup

The stack (VSS backend, DataEngine, VastDB, GPU models) is pre-deployed. Credentials come from environment variables
(from the team config file); none are stored in this repo.

| Variable | Purpose |
|---|---|
| `INGRESS_URL`, `USERNAME`, `PASSWORD` | VSS retrieval API (JWT login) |
| `S3_ENDPOINT`, `ACCESS_KEY`, `SECRET_KEY`, `VASTDB_BUCKET` | VastDB SDK (data VIP) |
| `GPU_BEARER_TOKEN`, optional `COSMOS3_REASON_URL` | Cosmos Reason on the GPU host |
| `WANDB_API_KEY`, `WANDB_TEAM`, `WANDB_PROJECT` | W&B Inference, Weave, W&B runs |
| `KUBECONFIG` | deploy to the team namespace |

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

## Reproduce

```bash
python explore_footage.py && python assess_footage.py     # find crossing footage (Phase 1)
python reingest.py                                         # optional: DataEngine re-ingest with the crossing prompt
python caption.py                                          # Cosmos Reason captions for the demo clips
python decide.py                                           # decisions for demo clips + 5 clips per street camera
python agent.py --n 4                                      # escalations + scorecard
python perceive.py                                         # per-frame tracks + moment decisions for the demo clips
python policy_eval.py                                      # segment-level vs frame-level policy replay
./deploy/deploy.sh                                         # web app at /app (open via the workshop App button)
python evaluate.py select                                  # pick 20 segments for blind labelling
#   label them at /app/label, then:
python evaluate.py score                                   # agreement, unsafe GOs, W&B run
```

## Inputs and outputs

**Inputs:** indexed segments from `vss-collection` (5-second H.264 clips grouped in 30-second chunks), their YOLO11
object counts, Cosmos Reason captions (our prompt), and for the scorecard sample the original index captions.

**Output table** `hackathon.crossing_decisions` (VastDB, one row per segment per run):
`run_id, segment_id, original_video, camera_id, location, segment_number, start_sec, end_sec, caption_source,
cosmos_tags (json), yolo_counts (json), decision, llm_decision, unclear, signal, hazards (json), reasons (json),
confidence, model, created_at`. The same rows are in `data/decisions.json`.

**Agent output:** `data/agent_output.json` with per-clip escalations `{at_sec, waited_s, action, reason}` and the scorecard.

**Current run:** 240 segments (30 demo segments with Cosmos captions + 210 segments from 35 clips across 7 street
cameras). Demo clips, with frame-level perception:

| Clip | What happens |
|---|---|
| `nyc_streets_cam-1` #22 | DON'T WALK with flowing traffic for 22 s; WALK lights at 21.6 s and the robot rolls at 22.6 s |
| `nyc_streets_cam-1` #29 | WALK from the start: crosses with the crowd, holds through the blinking countdown and the steady hand |
| `nyc_streets_cam-1` #21 | DON'T WALK for the whole clip: holds calmly, never starts against the signal |
| `sf_streets_cam-4` | no readable pedestrian signal: crosses in traffic gaps, longer gaps required when traffic is heavy |
| `sf_streets_cam-1` | busy corridor, cars and a van through the path: few, short gaps |

**Perception output** `data/perception.json`, per segment: `cfg` (calibration, `cross_s`), `obstacles` (Gemma 4 result,
`path`), `frames[]` at 5 fps (`v` vehicles `[x1,y1,x2,y2, 2=conflict|1=moving|0=stopped, class, vx, vy]`, `p` people,
`h` signal head `[x1,y1,x2,y2, walk|hand|dark]`) and `moments[]` (`t, d, why, sig, mv, pred, xp, lvl, quiet`).

## Evaluation

### Policy replay: segment-level vs frame-level (`policy_eval.py`)

The same robot is replayed on the 5 demo clips (150 s, 750 moments). It waits at the curb, starts on GO, crosses at
1.6 m/s, and the next robot arrives 1.6 s later. It runs twice: once acting on the segment LLM verdict (Cosmos caption →
Llama 3.3, one per 5 s), once on CrossWise's frame-level decisions. Both are checked in hindsight against the signal read
from the pixels and the vehicle tracks.

| | Segment LLM (5 s) | CrossWise frames (0.2 s) |
|---|---|---|
| Crossings completed | 6 | 8 |
| Starts against DON'T WALK / countdown | **3** | **0** |
| Vehicle in the path at start or within 2 s (robot must yield) | 3 | 3 |
| WALK lit → robot rolling (NYC #22) | 3.4 s | 1.0 s |
| Walk phase used (NYC #29, 13 s of WALK) | missed entirely | crossed |

Main findings from building it:

- The segment LLM read the round red *vehicle* light at SF crossing 4 as "don't walk" (0 crossings in a clip with 18 s of safe gaps).
- It said GO in NYC #21 and #22 while the robot's own signal showed a steady hand.
- At the NYC corner, two signal heads face the camera and show opposite phases. Only the one on the robot's crosswalk may be used.

CrossWise's 3 "vehicle in path ≤ 2 s" cases:

- a queued car creeping forward below the motion threshold (SF crossing 4);
- a stopped car at the frame edge pulling away (SF crossing 1);
- a tracking false alarm on a parked truck cut off by the frame edge, whose box jumped as pedestrians passed in front (NYC #29).

The robot yields mid-crossing in the replay.

### Human labels

Method: 20 segments, stratified by decision (8 GO, 6 WAIT, 6 NO-GO) and spread across all 7 cameras, selected with a
fixed seed (`evaluate.py select`). A human labelled each as **safe** or **unsafe** for a robot to *start* crossing,
**blind** to the model's decision (`/app/label`). GO counts as "safe"; WAIT and NO-GO count as "not safe now".
Unsafe GOs (GO where the human said unsafe) are the critical error and are reported separately.

<!-- EVAL_RESULTS -->

## Limitations

- **Recorded footage, decision support only.** No real-time control, no robot integration.
- **Small evaluation:** 20 labels from one labeller; GO is deliberately over-sampled, so agreement is not a population estimate.
- **Signal ambiguity:** a fixed camera often sees the *vehicle* light, not the pedestrian signal. The prompt treats a bare
  traffic-light colour as weaker evidence than a walk/don't-walk signal, but misreads remain.
- **Index captions are weaker than our crossing captions.** Scorecard clips use the original index captions, which rarely
  mention signal state (none in ~48% of segments) or turning vehicles, so turning hazards are under-counted outside the demo
  clips and NO-GO is over-represented ("vehicles moving over the crosswalk").
- **Per-camera calibration.** The crossing zone, corridor, crossing length and which signal head belongs to the robot's
  crosswalk are set by hand per fixed camera, from annotated frames. The signal reader needs the head to be at least 45 px
  tall; the SF cameras have no readable pedestrian head, so they use gap acceptance only. Only the five demo clips have
  per-frame tracks.
- **A fixed camera sees less than the robot would.** Cars entering the frame after a GO count against CrossWise in the
  policy replay ("vehicle in path ≤ 2 s"); a robot's own sensors would see them earlier. A stopped car pulling away
  cannot be predicted from motion.
- **Short clips:** chunks are 30 s, so "wait before GO" is a lower bound (streaks that never reach a GO count at their observed length).
- **Scorecard sample:** 5–6 clips per camera; the ranking is directional.
- **Re-ingest:** the shared DataEngine pipeline was replaying a backlog during the event, so crossing captions were produced by
  calling Cosmos Reason directly (same prompt, same model family) and stored in our own table rather than `vss-collection`.
- **Chunk videos are HEVC;** the app plays the H.264 segment files back to back instead.

## Privacy

Prompts instruct Cosmos Reason and the decision model never to describe individual people or licence plates. Original index
captions (which sometimes describe clothing) are never displayed or committed; only decisions, reasons, signal state,
hazards, and object counts are stored in `data/`.
