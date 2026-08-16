# Data Basics: What Is in a Sequence, and Where the USS Numbers Come From

Handy reference. Everything here is verifiable from the checked-in sample
`data/samples/110613_20190308_LB_XO2617_039_151540/` (bicyclestand, backward,
2019-03-08).

## Sequence identity

Folder name `110613_20190308_LB_XO2617_039_151540` decodes as:

| token | meaning |
|---|---|
| `110613` | FML sequence id (Bosch measurement-library primary key) |
| `20190308` | recording date YYYYMMDD -> the **session unit** used by the day-grouped split |
| `LB` | site code |
| `XO2617` | test vehicle id |
| `039` | run number within the day |
| `151540` | wall-clock HHMMSS |

Sequences were staged and recorded on the FML project `PLAT_USS`
(FML paths: `AutDevData\PLAT_USS\sequences\<id>\...`).

## Files per sequence

| file | what | who reads it |
|---|---|---|
| `<name>.json` | FML metadata: speeds, temperature, software version, **tags** (Approach angle, Driving Direction, Scene, Weather, **Sensor Type**, Label Objects) | nb01 inventory; direction -> camera choice |
| `<name>_Front.avi.mp4` / `_Rear.avi.mp4` | bumper camera video, 960x640 (webm originals converted by the downloader) | frames for labeling/training |
| `<name>.mf4` | full vehicle bus recording (ASAM MDF4, ~90 MB). Camera channels `WebCam`(front)/`WebCam3`(rear) pair every video frame's PTS to bus master time = exact alignment. Also `sigPDC_Zone*_Distance` (cm, ~1.2 m range) and `MAP_ObjBuff_elm*_LastDetDist` (mm, ~5 m range) | nb10/11 alignment + distance |
| `<name>_PAS_m_label_V1/V2.csv/.xlsx` | Bosch label files: staged object type + class per row (`NAMEoFFILE` families). V2 is what we use. NOTE: its `distance` field is a spec constant (2000 mm) - never a measurement | nb01 taxonomy; sequence-level truth |
| `VIDEO_USS_TOPVIEW.mp4/.jpg` | Bosch tool render of the USS map top view | visual sanity only |
| `CusReplay/<jobid>/<name>_PFDF4_DAQ_MAP_Classifier_Monitor_Normed.csv.xz` | **production height classifier monitor** (see below) | nb14/15 fusion; the USS baseline |
| `extracts/*.dat.bz2`, `MDF4_DAT_LOG/*.log` | ADTF raw extracts + replay pre-check logs | not used |

## Sensor

Metadata tag `Sensor Type: 12x6.5` on the sample = Bosch USS generation
naming (12 sensors on the vehicle, 6.5 = sensor variant/generation).
Vehicle software `MKS_CP_1.195.1.6__SIP6.15.10_PRE2.2` = production parking
stack build. Corpus-wide sensor-type distribution: run the scan cell in
`docs/data_basics.md` -> "Sensor scan" (below) - the nb01 inventory did not
capture this tag.

## What "CusReplay" is

Customer Replay = Bosch's resimulation tool. It takes the recorded raw sensor
stream (`.mf4` / `PFDF4.dat`) and re-runs the **production USS software**
offline, exactly as it executes in the vehicle, dumping internal module
signals that the vehicle bus never exposes. The `<jobid>` folder is one
replay job. `MAP_Classifier_Monitor_Normed` is the monitor of the height
classifier module: one row per processing cycle per tracked object.

Columns (all probabilities 0..255, normalized):

| column | meaning |
|---|---|
| `CycleNr` | processing cycle |
| `MAP_ClassifierMonitorNormed_ObjID` | tracked object id in the USS map |
| `..._ObjDist` | distance to that object (mm); >0 = a real reading |
| `..._ExistProb` | existence probability |
| `..._ClassProbHigh` | **P(object is high)** - the production height decision |
| `..._HeightProb` | related height confidence |
| `..._ObjP1/P2_X/Y` | object corner points in the map |

Sample: 566 cycles, 560 with a tracked object, dominant object #3 tracked
for 188 cycles.

## How the USS accuracy number (62.9% / 70.5%) is computed

We built NO classifier. We score the production classifier's own output:

1. Read `Classifier_Monitor_Normed.csv.xz`, keep rows with `ObjDist > 0`.
2. Take the **dominant tracked object** (most cycles) = the staged target.
3. Median of its `ClassProbHigh / 255` over all its cycles.
4. Threshold at **0.5** (production operating point) -> `high` / `low`.
5. Truth = staged object type from metadata -> its height bin (Low < 25 cm).
6. Accuracy = fraction of sequences where step 4 == step 5.

Code: `src/vision_uss_research/alignment/uss_class.py`
(`read_classifier_monitor`, `uss_sequence_decision`).

Sample walk-through: `ClassProbHigh` starts near 0 (far), crosses 0.5
around cycle 2550, settles ~0.93; median 0.93 -> HIGH; staged bicyclestand
= High -> correct. Figure: `paper/figures/data_walkthrough.png`.

Numbers: 62.9% on the 887-sequence joint-decision set (sequence-random
split); 70.5% on the 1,154-sequence day-grouped set. Same procedure, different
test populations. Median aggregation is deliberately generous to USS (the
strongest per-sequence reading the deployed system supports).

## Sensor scan (run on Colab, no GPU)

```python
# corpus-wide Sensor Type + software version from the FML metadata jsons
import json, collections
from pathlib import Path
sensor, sw = collections.Counter(), collections.Counter()
missing = 0
for d in sorted(p for p in RAW_ROOT.iterdir() if p.is_dir()):
    js = list(d.glob("*.json"))
    if not js:
        missing += 1; continue
    try:
        meta = json.loads(js[0].read_text())
    except Exception:
        missing += 1; continue
    st = [t["name"] for t in meta.get("tags", [])
          if t.get("tagSubCategory", {}).get("name") == "Sensor Type"]
    sensor[st[0] if st else "untagged"] += 1
    sw[meta.get("softwareVersion", "?")] += 1
print("sequences scanned:", sum(sensor.values()), "| no/unreadable json:", missing)
print("Sensor Type:", dict(sensor.most_common()))
print("software versions:", len(sw), "| top:", sw.most_common(3))
```
