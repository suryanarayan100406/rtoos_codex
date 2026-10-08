# DRISHTI — Implementation docs

**Purpose:** Index for the implementation-facing documentation — the *build* docs, as opposed to the
evaluator-facing story in [`../`](../). Start here if you are implementing DRISHTI.

**Audience:** The build team and AI agents. The very first thing to read is the repo-root living context
file, [`../../AGENTS.md`](../../AGENTS.md).

**TL;DR**
- This build is **footage-in → full 3D map, offline, ground-only** (no drone/edge, no live path) on
  **modest + free compute** (Intel Ultra 7 155H + free cloud T4). Scope rationale:
  [`../../AGENTS.md` §2](../../AGENTS.md).
- Read in order: **AGENTS.md → PRD → Implementation Plan → System Design.**
- The authoritative pipeline/model registry lives in
  [`../_internal/CANONICAL-ARCHITECTURE-SPEC.md`](../_internal/CANONICAL-ARCHITECTURE-SPEC.md); these
  docs realize it offline and note any deltas.

## Documents

| Doc | What it covers |
|-----|----------------|
| [`../../AGENTS.md`](../../AGENTS.md) | **Living context (read first):** scope, constraints, principles, repo layout, status, decision log, rules for agents |
| [00 · PRD](00-PRD.md) | Product requirements: users, functional (F#) + non-functional (N#) requirements, outputs/formats, success metrics, acceptance criteria |
| [01 · Implementation Plan](01-IMPLEMENTATION-PLAN.md) | All seven phases (0–6) planned: goals, stages, tasks, local/cloud compute placement, deliverables, exit criteria |
| [02 · System Design](02-SYSTEM-DESIGN.md) | Code architecture: repo layout, project-bundle artifact model, stage interface, config system, compute tiering, model registry |

## Reading paths

- **New AI agent / contributor:** AGENTS.md → this index → PRD → Plan → System Design.
- **"What do I build next?"** → [Implementation Plan](01-IMPLEMENTATION-PLAN.md) + the status table in
  [`../../AGENTS.md` §8](../../AGENTS.md).
- **"How is the code organized / where does this stage run?"** → [System Design](02-SYSTEM-DESIGN.md).

## Open questions / risks

Tracked per document (each ends with its own list) and, project-wide, in
[`../../AGENTS.md` §12](../../AGENTS.md). The dominant near-term risk is acquiring a **representative
real dataset** to validate Phases 2–5 against.
