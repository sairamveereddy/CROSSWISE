# CrossWise: 2-minute live demo script

Open the app before you start: workshop.thecosmoslabs.com → **App** (team-48). Hard-refresh once so the video is warm.
Keep a second tab on the Weave traces (wandb.ai/vastdata/team-48/weave).

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | App header, robot logo, the yellow notice | "Sidewalk delivery robots get stuck at street crossings. CrossWise is a crossing copilot: for every moment at a crossing it says GO, WAIT, or NO-GO, with reasons and video evidence, and it escalates before a robot sits stuck. This is decision support on recorded footage; it doesn't drive a robot." |
| 0:15–0:45 | Featured tile **SF crossing 4** → click **Replay as live** | "This is real footage indexed by VAST DataEngine. Each 5-second moment was captioned by NVIDIA Cosmos Reason on CoreWeave GPUs with our crossing prompt, combined with YOLO detections, and decided by an LLM on W&B Inference. Watch the robot on the timeline: red light, hold at the curb. WAIT, WAIT, WAIT…" |
| 0:45–1:00 | Banner fires at 20 s | "Twenty seconds without a safe gap: the agent escalates to a remote operator and suggests the easiest nearby crossing from our scorecard. N is configurable right here." |
| 1:00–1:20 | Click tile **NYC crossing 1**, then a segment; point at Why / Evidence | "Different situation: don't-walk signal plus cones and a barrier. Every decision shows why: the signal, hazards, YOLO counts, and the Cosmos caption. Because the crosswalk itself is obstructed, the action here is **reroute**, not wait. Unclear evidence always becomes WAIT, and a code guardrail can never let a GO stand if Cosmos saw a turning vehicle or a blocked crosswalk." |
| 1:20–1:35 | Tile **SF crossing 1** (GO appears) | "Here the robot does get GOs between turning traffic and a truck stopped in the crosswalk, so no escalation." |
| 1:35–1:50 | Scroll to **scorecard**, then type a query in search | "Across 7 street cameras we rank crossings from easiest to hardest: share of GO time, wait before a GO, turning hazards. That feeds route planning. And search runs on the VAST index with Cosmos Embed, to find any moment like 'truck blocking the crosswalk'." |
| 1:50–2:00 | Weave tab (or say it) | "Every decision is traced in Weave, results live in a VastDB table, and we evaluated against 20 blind human labels: <AGREEMENT>% agreement with <UNSAFE_GO> unsafe GOs. VAST, NVIDIA, CoreWeave, W&B, built with Cursor. Thank you." |

**Fallbacks:** if video stalls, click a timeline segment (each is a separate 5 s file). If search is slow, skip it; the
core is the replay, the timeline with reasons, and one escalation firing.
