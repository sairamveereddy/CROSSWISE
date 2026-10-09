# CrossWise: 2-minute live demo script

Open the app before you start: workshop.thecosmoslabs.com → **App** (team-48). Hard-refresh once so the video is warm.
Keep a second tab on the Weave traces (wandb.ai/vastdata/team-48/weave).

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | App header, robot logo, the notice | "Sidewalk delivery robots lose minutes at crossings: they spin at the curb, miss the walk phase, or start into traffic. CrossWise is a crossing copilot. Five times a second it says GO, WAIT or NO-GO with evidence. This is decision support on recorded footage; it doesn't drive a robot." |
| 0:15–0:50 | Tile **New York · #22** → **Replay as live**, Detections on | "Real footage indexed by VAST. The boxes are YOLO11 detections VAST stores for every frame; we track them, remove the handheld camera's shake and predict each car 1.5 s ahead. The box top-left is the pedestrian signal *for the robot's crosswalk*: CrossWise reads it from the pixels. DON'T WALK, cars flowing: the robot holds calmly, no spinning. Watch 21 s: WALK lights… and the robot rolls one second later, steering right of the cones Gemma 4 found on the crosswalk." |
| 0:50–1:10 | Tile **New York · #29** | "Next cycle. WALK from the start: it crosses with the crowd. Then the hand starts blinking: the countdown. Too late to start, so it waits for the next WALK instead of getting caught mid-road." |
| 1:10–1:25 | Tile **San Francisco · Crossing 4** | "No readable pedestrian signal here, so it switches to gap acceptance: GO when the path has been quiet long enough. 0.6 s in light traffic, 2 s when traffic is heavy. It crosses small gaps immediately and waits out the busy stretch." |
| 1:25–1:45 | Scroll to **Impact** | "Same robot, two brains. One LLM verdict per 5 s segment from Cosmos captions started against DON'T WALK three times, missed a whole 13-second walk phase and was 3.4 s late on WALK. Frame-level CrossWise: zero starts against the signal, more crossings, 1 s from WALK lit to rolling." |
| 1:45–2:00 | Scorecard / search, Weave tab | "We also rank crossings for route planning and search the archive with Cosmos Embed. Every LLM and Gemma call is traced in Weave and decisions land in a VastDB table. VAST, NVIDIA, CoreWeave, W&B, built with Cursor. Thank you." |

**Fallbacks:** if video stalls, click a timeline segment (each is a separate 5 s file) or drag on the strip under the
bars. If search is slow, skip it. The core is NYC #22 (hold on DON'T WALK, roll on WALK) and the Impact numbers.
