# Paper Plan: Layout, Venue, Timeline

Status 2026-08-02. All experiments complete (see `v2_open_set_approach.md`);
this is the writing plan. Working title:

> **Drive-Over or Stop? Open-Set Traversability Detection for Near-Field
> Parking from Automatically Labeled Camera-Ultrasonic Data**

## Venue decision

| venue | deadline | notes |
|---|---|---|
| **WACV 2027, Round 2** | **2026-08-28** (suppl. 08-30), decisions 10-09 | Single-shot (no rebuttal). WACV explicitly welcomes applications papers; algorithms/applications tracks. 26 days out. |
| **IEEE IV 2027** (Perth, Jun 2027) | **2026-11-15** | Best audience fit: parking, USS, production-stack comparison. Notification 2027-01-15. |
| ITSC 2027 (Boston, Sep 2027) | ~Feb 2027 (unannounced) | Fallback; same community as IV. |
| WACV workshop / CVPR-W 2027 | winter | Fallback if main tracks reject. |

**DECIDED 2026-08-02: WACV 2027 Round 2 (Aug 28), with IV 2027 (Nov 15) as
the pre-planned fallback.** Note CVPR 2027's deadline (~Nov 13) collides with
IV's; WACV reviews arrive Oct 9, so a WACV reject still leaves both November
options open. CVPR main is a poor fit for this paper (applications profile,
proprietary data, no public benchmark) - IV is the November default unless
the WACV reviews argue otherwise. Rationale: every number is already in hand, so 26
days of pure writing is feasible; the CV-side contributions (SAM3-as-annotator
at scale, trained open-set buckets, leave-object-out protocol) are what WACV
values; and a WACV reject returns reviews by Oct 9 - five weeks before the IV
deadline, enough to revise and resubmit with almost no wasted work. The only
schedule risk is the one open experiment (holdout rotation, below) - it runs
in week 1 either way.

Formats: WACV 8 pages + references (their template); IV 6-8 pages IEEE
format. Write WACV-first; an IV port is mostly compression.

## The story (three claims, one sentence each)

1. **Auto-labeling**: a SAM3 text-prompt + corridor-prior + tracker pipeline
   labels ~8k real parking sequences (~136k boxes) with ~170 human clicks,
   validated against a human-clicked gold test set.
2. **Open-set traversability**: trained "other" buckets generalize the
   Low/High (drive-over) decision to never-seen obstacle types, measured
   honestly with leave-object-out holdouts.
3. **Camera vs production USS**: on gold sequences the camera detector (93.6%)
   corrects the production ultrasonic height classifier (62.9%) on **93% of
   its errors**; failure modes are complementary (USS wins on flat/low,
   camera on tall/vertical), and learned fusion confirms the camera is
   near-ceiling. Two negative findings (geometric height is fundamentally
   ambiguous for flat objects; fusion gain is marginal) are reported as
   findings, not buried.

## Section layout (8 pages)

1. **Introduction** (~1 pp). The drive-over decision in near-field parking;
   USS is production reality but height classification is weak; cameras see
   what USS cannot. Contributions list = the three claims. Teaser figure.
2. **Related work** (~0.75 pp). (a) foundation-model auto-labeling /
   SAM-as-annotator; (b) open-set & unknown-aware detection; (c)
   camera-ultrasonic / near-field parking perception; (d) monocular metric
   height & its ambiguities.
3. **Dataset & problem** (~1 pp). Test-track corpus: ~8.2k staged approach
   sequences, front/rear bumper cameras, synchronized MF4 bus data, per-cycle
   production USS classifier logs; 18+ obstacle types, Low = <25 cm
   drive-over. MF4-video alignment (exact PTS pairing). State the domain
   honestly: single facility, single vehicle, 2019 imagery.
4. **Auto-labeling pipeline** (~1.25 pp). SAM3 text prompts -> approach-
   corridor instance prior -> video-tracker propagation; clicks only for 3
   hard classes; per-class QA gates. Pipeline figure + label-stats table.
5. **Open-set detector** (~1 pp). 6 named classes + low_other/high_other
   buckets trained on 12 diverse types; deterministic class->Low/High map;
   leave-object-out protocol.
6. **Experiments** (~2 pp). E1 label quality vs gold clicks; E2 detection
   (mAP50 0.79; matched Low/High 99.8%; per-sequence 93.6%); E3 open-set
   holdout table (cone + rotations); E4 camera vs USS vs fusion (the headline
   table + complementarity figure); E5 geometric-height negative result
   (motivates the learned detector).
7. **Limitations** (~0.4 pp). Domain; per-sequence protocol (staged single
   target); min-over-slots USS distance; monocular flat-object ambiguity;
   fusion marginality.
8. **Conclusion** (~0.25 pp).

## Figures & tables (all data already exists)

| item | content | source |
|---|---|---|
| Fig 1 teaser | approach frames + detections + USS distance curve + both decisions | nb09 sim export / nb11 overlay |
| Fig 2 pipeline | SAM3 -> corridor -> propagate -> QA -> YOLO diagram | draw |
| Fig 3 qualitative | per-class detection gallery incl. buckets on held-out cone | nb08 |
| Fig 4 complementarity | per-class camera vs USS accuracy bars (mirror image) | nb14 per-class table |
| Fig 5 height ambiguity | flat-object box-top geometry sketch + est-vs-true scatter | nb12/13 |
| Tab 1 dataset | sequences/classes/boxes/clicks stats | nb01 + label runs |
| Tab 2 detection | per-class AP + Low/High accuracy | nb07 |
| Tab 3 open-set | holdout rotations: bucket recall + Low/High on unseen type | nb07 reruns |
| Tab 4 fusion | USS / camera / rule / meta bal-acc + rescue rates | nb14/15 |

## Remaining work items

1. **Holdout rotation (the one open experiment, week 1).** Currently only
   cone. Configs are ready: `dataset_v2_holdout_bollard.yaml` /
   `dataset_v2_holdout_bush.yaml` (identical to dataset_v2 except the held-out
   type; ~1 GPU day per rotation). Recipe per rotation, e.g. bollard:
   - nb05: `CONFIG_NAME = "dataset_v2_holdout_bollard"` (seconds - manifest only)
   - nb06: `DATASET_RUN = "dataset_v2holdbollard"`,
     `TRAIN_TAG_OVERRIDE = "v1_holdbollard"`
   - nb07: `MODEL_RUN = "train_v1_holdbollard"`, `EVAL_TAG = "holdbollard"`
   The tag knobs keep rotation runs fully separate from the baseline
   `train_v1` / `eval_test` runs. Table 3 rows = cone (existing) + bollard +
   bush; report bucket recall and Low/High accuracy on the held-out type.
   Pre-registered risk in `v2_open_set_approach.md`.
2. **Writing** (below).
3. Optional, only if time: scene-tag stratification of bucket recall
   (anti-shortcut control); investigate 2,138 missing MF4s (does not block -
   fusion population is defined and reported).

## Timeline to Aug 28 (26 days)

| dates | milestone |
|---|---|
| Aug 3-8 | Launch holdout rotations on Colab. Meanwhile: paper skeleton in LaTeX (WACV template), Tab 1/2/4 filled from existing runs, Fig 4/5 scripted. |
| Aug 9-15 | Full draft of sections 3-6 (results-first). Fig 1-3 produced from nb08/09 artifacts. Rotation results -> Tab 3. |
| Aug 16-21 | Intro, related work, limitations. Internal full read; numbers cross-checked against run.json provenance. |
| Aug 22-26 | Polish pass, co-author review, supplementary (extra galleries, per-class tables, reproducibility statement). |
| Aug 27-28 | Buffer + submit (suppl. due Aug 30). |

Decision gate Aug 15: if rotations look bad (bucket recall ~0 on a second
held-out type), reframe claim 2 to the cone result + honest discussion, or
retarget IV 2027 with three extra months. Either way nothing is wasted.

## Data-availability statement

Raw data is proprietary (Bosch/Daimler test track). The paper's citable
contributions that survive that: the leave-object-out open-set protocol, the
auto-labeling recipe (clicks-not-boxes budget), the alignment method
(`mf4_video_alignment.md`), and all per-class result tables. State "data
cannot be released; code and protocol are described for reproduction on
comparable corpora" - standard for industrial AD papers at IV/ITSC and
accepted at WACV applications track.
