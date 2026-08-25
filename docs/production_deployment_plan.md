# Production Deployment Track: Camera Low/High Assist for the USS Stack

Status 2026-08-21. Side project, separate from the WACV paper (submission
#2660, decisions Oct 9). Goal: take the trained YOLO11s Low/High detector
from research artifact to a candidate assist function in the production
parking pipeline.

## What we have (validated) vs. what production needs

The paper established, honestly and with artifacts:

| validated | NOT validated (the production gap) |
|---|---|
| Decision correctness when the obstacle is detected: 99.3% from the first observed frame, false-Low 0.5% early / 6.5% overall | Runtime, latency, memory on any target hardware |
| Camera corrects the production USS classifier on 93-96% of its errors; result survives a session-independent split | Causal *deployed* decision logic (streaming vote, hysteresis, handoff) - our causal analysis was an offline replay |
| Complementarity map: camera wins tall/thin, USS wins flat/low + vegetation | Behavior in multi-obstacle scenes (corpus is staged single-obstacle) |
| Coverage limit quantified: camera abstains on ~4-24% depending on population; out-of-support types (bush) invisible | In-the-wild domain: one facility, one vehicle, 2019-2020 imagery |
| Label pipeline + evaluation harness, fully reproducible | **Production camera domain**: corpus uses dev webcams (960x640 bumper mounts); production vehicles carry fisheye surround-view cameras - a real domain shift |
| | Functional-safety case (ISO 26262 / SOTIF): the model is NOT a validated safety function |

## Deployment posture (the one big design decision)

The evidence supports exactly one initial role: **assist / plausibilization
signal to the USS height classifier, asymmetric and conservative**:

- Camera may RAISE the stack's confidence toward High (stop) - the safe
  direction; its false-High rate on Low obstacles is 6.1%, cost = comfort.
- Camera overriding USS toward Low (drive over) is the dangerous direction
  (false-Low = collision). The 93%-rescue result includes many
  USS-false-High -> camera-correct-Low cases (comfort wins), but enabling
  that direction needs a false-Low budget agreed with safety - phase 5,
  not phase 1.
- Camera abstention (no detection) must be a first-class output: on
  abstain, the USS decision stands unchanged. The paper's coverage numbers
  make this the normal case for out-of-support obstacles.

Start in **shadow mode**: compute and log the camera decision alongside the
live stack, influence nothing, compare offline. Exactly the CusReplay
philosophy, in reverse.

## Phases

**P0 - scoping (talk before build).** Decide with the platform team:
target ECU / SoC and compute budget; which camera feeds are accessible
(surround-view fisheye? which resolution/fov?); where in the stack the
assist signal would enter (before/after the height classifier); who owns
the safety assessment. Output: one-page interface + constraints memo.
BLOCKING for everything below except P1a.

**P1 - packaging + runtime benchmark.**
- P1a: export train_v1 (and daysplit variant) to ONNX; verify numerical
  parity on the gold test set (same 93.6% within tolerance).
- P1b: quantize (FP16/INT8) + benchmark on the P0 target (or a proxy like
  Jetson/Snapdragon dev kit): FPS, latency, memory, accuracy delta.
  Fills the paper's own "no runtime evaluation" gap - also feeds a
  camera-ready update.

**P2 - causal decision module.** Turn the offline replay into a streaming
component: per-frame inference -> running majority vote with hysteresis ->
tri-state output (Low / High / abstain) + confidence + "obstacle leaving
FOV" flag for the USS handoff. Pure software; testable against recorded
sequences with the existing harness (frame_predictions.csv logic).

**P3 - shadow-mode resim.** Run the P2 module over held-out and NEWLY
recorded sequences next to CusReplay output; produce the same
joint-decision comparison tables the paper uses. Acceptance gate: matches
paper numbers on old data; measures (rather than assumes) performance on
new recordings.

**P4 - domain adaptation.** The likely hard part:
- Fisheye/production-camera data: collect or convert (the labeling pipeline
  transfers as-is - SAM3 does not care about the camera); fine-tune and
  re-run the P3 harness.
- Recovered sequences from the FML retry (internal/ scripts) + any 2021+
  recordings widen the support.
- Re-run leave-object-out on the new domain: the absorption/invisibility
  findings define what the assist must NOT be trusted for.

**P5 - safety case.** SOTIF-style analysis with the safety team: false-Low
budget, abstention semantics, degradation behavior (dirty lens, night,
rain), misuse cases. Only after this can the Low-override direction be
considered. The paper's limitations section is literally the starting
hazard list.

## Immediate next steps

1. Ali: identify the platform/stakeholder for P0 (which production line,
   which ECU, camera access).
2. Claude: P1a ONNX export + parity check notebook (no target hardware
   needed - runs on Colab; ~1 session of work). Can start immediately.
3. Keep the paper population frozen; everything here uses new run ids
   under a `prod_` prefix so research artifacts stay untouched.

## Non-goals (for now)

Standalone camera-only parking function; metric height (paper: ill-posed
for flat obstacles); learned camera-USS fusion (paper: no measurable gain);
any claim of unknown-obstacle detection (paper: buckets do not deliver it).
