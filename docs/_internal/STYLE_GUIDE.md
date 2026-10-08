# Documentation Style Guide (authoring contract)

Every document in this project MUST follow this guide so the set reads as one coherent submission.

---

## 1. Project identity

- **System codename:** **DRISHTI** (Sanskrit *dṛṣṭi*, "vision / sight").
  Backronym: **D**rone-based **R**eal-time **I**maging for **S**ingle-pass **H**igh-fidelity
  **T**errain **I**ntelligence.
  - Alternate codename on record (if the team prefers): **TRINETRA** ("three-eyed / all-seeing").
- **One-line pitch:** *"One pass. The whole picture — a georeferenced, measurable 3D model of
  everything the drone flew over, built while it flies."*
- Always refer to the system as **DRISHTI** in prose (not "our system", not "the app").

## 2. Audience & tone

- Primary audience: **NTRO technical evaluators** (defense/intel, mapping, systems engineers) and
  hackathon judges. Secondary: our own build team.
- Tone: **technical, confident, quantitative, and honest.** No marketing fluff, no unverifiable
  superlatives. Prefer numbers, named methods, and concrete interfaces over adjectives.
- When you state a capability, state its **operating envelope** (when it holds) and its **failure
  mode** (when it doesn't). Evaluators trust documents that admit limits.

## 3. Honesty policy (non-negotiable — see PROBLEM_STATEMENT.md §4)

- "Real-time" → **near-real-time edge preview + minutes-scale ground refinement.** Never imply a full
  textured mesh in hard real-time on the UAV.
- "Never fails" → **graceful degradation, no single point of failure, always emits a best-effort model
  + uncertainty report.**
- "Metric without GCPs" → accurate via fused GNSS/RTK/PPK+IMU+visual scale, with
  **sensor-configuration-dependent** accuracy numbers.
- Every learned (neural) output must be described as producing a **confidence/uncertainty** signal.

## 4. Canonical terminology (use these exact terms)

| Use this | Not this |
|----------|----------|
| UAV / drone | "quadcopter" (unless specific) |
| keyframe | "snapshot", "still" |
| VIO (Visual-Inertial Odometry) | "visual tracking" |
| GNSS (GPS/GLONASS/Galileo/BeiDou) | "GPS" except when referring to plain GPS input |
| pointmap (feed-forward per-pixel 3D) | "depth image" (that is depth map) |
| TSDF fusion | "voxel merge" |
| 3DGS (3D Gaussian Splatting) | "splatting" alone |
| georeferencing | "geolocating" |
| GSD (Ground Sampling Distance) | "resolution" |
| DSM / DTM | "elevation map" (be specific) |
| CRS (Coordinate Reference System) | "map projection" alone |
| GCP (Ground Control Point) | — |
| Edge Tier / Ground Tier / Cloud Tier | "onboard/offboard" (use the tier names) |

Expand every acronym on first use in each document.

## 5. Architecture is defined ONCE, elsewhere

The authoritative pipeline (tiers, stages S0–S10, data flow, model choices, interfaces) lives in
**`docs/_internal/CANONICAL-ARCHITECTURE-SPEC.md`**. Read it before writing. Do **not** invent
alternative stage names, tier names, or model choices — cite the spec. If you believe the spec is
wrong, note it in your document's "Open questions" section rather than contradicting it silently.

## 6. Formatting rules

- Markdown, GitHub-flavored. One `#` H1 title per file; sentence-case headings.
- Use **tables** for comparisons/decisions/specs; use fenced code blocks for interfaces, message
  schemas, CLI, and data formats.
- Diagrams: use **Mermaid** fenced blocks (```` ```mermaid ````) — they render on GitHub and in the
  pitch. Prefer `flowchart`, `sequenceDiagram`, `graph`. Keep node labels short.
- Cross-reference other docs by relative path, e.g. `see [Theory](01-THEORY.md)`.
- Wrap prose at a readable width; keep paragraphs short. Lead each major section with a 1–2 sentence
  summary.
- Every document starts with: an H1 title, a one-line **Purpose** blurb, an **Audience** line, and a
  short **TL;DR** (3–6 bullets). End with **Open questions / risks** and, where relevant, a
  **Glossary** or **Further reading**.
- Numbers: give ranges and state assumptions (altitude/AGL, sensor, CRS). Mark any figure that is a
  design target vs. a measured result.

## 7. Consistency anchors (repeat these verbatim where relevant)

- **Three tiers:** Edge Tier (on-UAV, NVIDIA Jetson Orin) · Ground Tier (station/server GPU) · Cloud
  Tier (optional, scale-out & serving).
- **Two output paths:** *Live path* (S0–S5, near-real-time coarse map) and *Refine path* (S6–S10,
  minutes-scale metric textured model).
- **Reliability spine:** every stage emits confidence; every stage has a fallback; nothing hard-fails.
- **Metric spine:** tightly-coupled GNSS(+RTK/PPK)+IMU+visual factor graph gives scale & georeference
  without GCPs.

## 8. What to avoid

- Do not duplicate large content across documents — link instead. (Some intentional repetition of the
  four "consistency anchors" above is fine and encouraged.)
- Do not cite tools/models the architecture spec didn't adopt without flagging them as alternatives.
- Do not claim benchmark numbers as *our measured results*; label external numbers as "reported by
  the method's authors" and our numbers as "target" until measured.
