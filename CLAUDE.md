# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is currently in the **planning phase** — there is no source code, build system,
or test suite yet. It contains planning documents and a small reference-paper library. Do not
assume any code architecture exists; read [plan.md](plan.md) and
[elaborate_plan.md](elaborate_plan.md) to understand what is being built before proposing an
implementation.

## Project goal

Real-time AI-Kalman state estimation running live on a Crazyflie drone, holding/estimating
altitude (`z`) accurately during flight **without a direct vertical height sensor** (the onboard
ToF/z-ranger withheld from the estimator), by fusing a neural-network-based estimator with a
Kalman filter using observability-informed, time-varying measurement trust. This is a direct
application of the **Augmented Information Kalman Filter (AI-KF)** method from
[references/summaries/bens_epic_paper.md](references/summaries/bens_epic_paper.md)
(Cellini et al. 2025, arXiv:2511.08766) to Crazyflie hardware.

Key insight carried over from that paper: altitude `z` is only observable from optic-flow +
accelerometer sensing when there is **non-zero horizontal acceleration** (speed-up/slow-down
bursts) — not from hovering or constant-velocity/vertical motion. This drives the whole
data-collection and flight-test design (see elaborate_plan.md §2–3 for the full
paper-to-hardware mapping and phased roadmap).

Planned architecture (not yet implemented): a Raspberry Pi 5 acts as a ground/edge station, not
an onboard companion computer — the Crazyflie streams IMU + optical flow over Crazyradio PA to
the RPi 5 (`cflib`), which runs the AI-KF pipeline and pushes the fused altitude estimate back as
an external measurement via the same CRTP path the firmware uses for mocap/Lighthouse/Loco
position sources. The onboard ToF sensor stays logged throughout as ground truth but is excluded
from the estimator's inputs. See elaborate_plan.md §3–4 for the full phased roadmap (Phase 0
environment setup through Phase 8 live flight demo) and the risk register (§5) and open
decisions (§6) before starting new work — check which phase is current before assuming later
work is unblocked.

## Working with the reference library

`references/` holds structured summaries of papers relevant to this project, built with the
`digest-paper` skill (`.claude/skills/digest-paper/SKILL.md`).

- `references/INDEX.md` — one paragraph per paper, links to the full summary. Read this first to
  see what's already been digested.
- `references/summaries/*.md` — full structured summary per paper (Key Finding, Methods, Main
  Results, Limitations, Relevance to This Project, etc.), one file per paper, named by slug
  (e.g. `bens_epic_paper.md`).
- To add a new paper to the library, use the `digest-paper` skill rather than summarizing ad hoc
  — it enforces a consistent template and appends to INDEX.md without disturbing existing
  entries. It reads this CLAUDE.md to fill in each summary's "Relevance to This Project" section,
  so keep the Project goal section above current.
- Source PDFs (e.g. `Bens_epic_paper.pdf`) live at the repo root, not in `references/`.

## Other local tooling

- `.claude/skills/make-cheatsheet/` — generates a validated quick-reference doc for a Python
  package (e.g. `cflib`, `pybounds`) once those dependencies are actually in use; it inspects the
  installed package via `dir()`/`inspect` rather than relying on memory, and validates the result
  with `validate_cheatsheet.py`. Not yet exercised in this repo since no dependencies are
  installed.
