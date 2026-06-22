from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class SequenceRecord:
    sequence_id: str
    sequence_dir: str
    front_video: str | None
    rear_video: str | None
    topview_video: str | None
    front_image: str | None
    rear_image: str | None
    topview_image: str | None
    label_v1_csv: str | None
    label_v2_csv: str | None
    label_v1_xlsx: str | None
    label_v2_xlsx: str | None
    metadata_json: str | None
    mf4_file: str | None
    pfdf4_pure_bz2: str | None
    is_poc_ready: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return str(path.resolve().relative_to(root.resolve()))


def _pick_first(paths: Iterable[Path]) -> Path | None:
    items = sorted(paths)
    return items[0] if items else None


def _pick_video(paths: Iterable[Path]) -> Path | None:
    items = list(paths)
    if not items:
        return None

    priority = {".mp4": 0, ".webm": 1, ".avi": 2}
    items = sorted(items, key=lambda p: (priority.get(p.suffix.lower(), 99), str(p)))
    return items[0]


def _find_files(seq_dir: Path) -> dict[str, Path | None]:
    all_files = [p for p in seq_dir.iterdir() if p.is_file()]

    front_videos = [
        p for p in all_files
        if "front" in p.name.lower() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    rear_videos = [
        p for p in all_files
        if "rear" in p.name.lower() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    topview_videos = [
        p for p in all_files
        if ("topview" in p.name.lower() or "uss" in p.name.lower())
        and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    front_images = [
        p for p in all_files
        if "front" in p.name.lower() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    rear_images = [
        p for p in all_files
        if "rear" in p.name.lower() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    topview_images = [
        p for p in all_files
        if ("topview" in p.name.lower() or "uss" in p.name.lower())
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_v1_csv = [
        p for p in all_files
        if "pas_m_label_v1" in p.name.lower() and p.suffix.lower() == ".csv"
    ]
    label_v2_csv = [
        p for p in all_files
        if "pas_m_label_v2" in p.name.lower() and p.suffix.lower() == ".csv"
    ]
    label_v1_xlsx = [
        p for p in all_files
        if "pas_m_label_v1" in p.name.lower() and p.suffix.lower() == ".xlsx"
    ]
    label_v2_xlsx = [
        p for p in all_files
        if "pas_m_label_v2" in p.name.lower() and p.suffix.lower() == ".xlsx"
    ]

    metadata_json = [p for p in all_files if p.suffix.lower() == ".json"]
    mf4_files = [p for p in all_files if p.suffix.lower() == ".mf4"]
    pfdf4_files = [
        p for p in all_files
        if p.name.lower().endswith(".pure.bz2") or "pfdf4" in p.name.lower()
    ]

    return {
        "front_video": _pick_video(front_videos),
        "rear_video": _pick_video(rear_videos),
        "topview_video": _pick_video(topview_videos),
        "front_image": _pick_first(front_images),
        "rear_image": _pick_first(rear_images),
        "topview_image": _pick_first(topview_images),
        "label_v1_csv": _pick_first(label_v1_csv),
        "label_v2_csv": _pick_first(label_v2_csv),
        "label_v1_xlsx": _pick_first(label_v1_xlsx),
        "label_v2_xlsx": _pick_first(label_v2_xlsx),
        "metadata_json": _pick_first(metadata_json),
        "mf4_file": _pick_first(mf4_files),
        "pfdf4_pure_bz2": _pick_first(pfdf4_files),
    }


def _looks_like_sequence_dir(seq_dir: Path) -> bool:
    found = _find_files(seq_dir)

    evidence_count = sum(
        value is not None
        for value in [
            found["front_video"],
            found["rear_video"],
            found["topview_video"],
            found["front_image"],
            found["rear_image"],
            found["topview_image"],
            found["label_v1_csv"],
            found["label_v2_csv"],
            found["label_v1_xlsx"],
            found["label_v2_xlsx"],
            found["metadata_json"],
            found["mf4_file"],
            found["pfdf4_pure_bz2"],
        ]
    )

    return evidence_count >= 3


def discover_sequence_dirs(raw_root: Path, show_progress: bool = True) -> list[Path]:
    raw_root = raw_root.resolve()

    print("Counting directories...")
    all_dirs = [p for p in raw_root.rglob("*") if p.is_dir()]
    print(f"Total directories found under raw root: {len(all_dirs)}")

    candidates: list[Path] = []

    iterator = all_dirs
    if show_progress:
        iterator = tqdm(all_dirs, desc="Scanning directories", unit="dir")

    for path in iterator:
        if _looks_like_sequence_dir(path):
            candidates.append(path)
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(candidates=len(candidates))

    candidates = sorted(set(candidates))
    final_dirs: list[Path] = []

    iterator2 = candidates
    if show_progress:
        iterator2 = tqdm(candidates, desc="Filtering nested candidates", unit="seq")

    for candidate in iterator2:
        has_child_sequence = any(
            other != candidate and candidate in other.parents
            for other in candidates
        )
        if not has_child_sequence:
            final_dirs.append(candidate)

    return sorted(final_dirs)


def build_manifest(raw_root: Path, show_progress: bool = True) -> list[SequenceRecord]:
    raw_root = raw_root.resolve()
    sequence_dirs = discover_sequence_dirs(raw_root, show_progress=show_progress)

    records: list[SequenceRecord] = []

    iterator = sequence_dirs
    if show_progress:
        iterator = tqdm(sequence_dirs, desc="Building manifest records", unit="seq")

    for seq_dir in iterator:
        found = _find_files(seq_dir)
        rel_dir = str(seq_dir.relative_to(raw_root))
        sequence_id = rel_dir.replace("\\", "__").replace("/", "__")

        is_poc_ready = all(
            [
                found["front_video"] is not None,
                found["topview_video"] is not None,
                found["label_v2_csv"] is not None,
            ]
        )

        records.append(
            SequenceRecord(
                sequence_id=sequence_id,
                sequence_dir=rel_dir,
                front_video=_rel(found["front_video"], raw_root),
                rear_video=_rel(found["rear_video"], raw_root),
                topview_video=_rel(found["topview_video"], raw_root),
                front_image=_rel(found["front_image"], raw_root),
                rear_image=_rel(found["rear_image"], raw_root),
                topview_image=_rel(found["topview_image"], raw_root),
                label_v1_csv=_rel(found["label_v1_csv"], raw_root),
                label_v2_csv=_rel(found["label_v2_csv"], raw_root),
                label_v1_xlsx=_rel(found["label_v1_xlsx"], raw_root),
                label_v2_xlsx=_rel(found["label_v2_xlsx"], raw_root),
                metadata_json=_rel(found["metadata_json"], raw_root),
                mf4_file=_rel(found["mf4_file"], raw_root),
                pfdf4_pure_bz2=_rel(found["pfdf4_pure_bz2"], raw_root),
                is_poc_ready=is_poc_ready,
            )
        )

    return records


def write_manifest_csv(records: list[SequenceRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [field.name for field in SequenceRecord.__dataclass_fields__.values()]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())