from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

VALID_RATIOS = ["0.15", "0.30", "0.45", "0.60", "0.75", "0.90"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    # Robust union of fields, so resumed runs with older rows do not lose new columns.
    fieldnames: list[str] = []
    seen: set[str] = set()

    preferred_order = [
        "sequence_id",
        "preferred_camera_rule",
        "preferred_camera_rule_reason",
        "driving_direction",
        "approach",
        "weather_tags",
        "scene_tags",
        "metadata_json",
        "label_objects_json",
        "target_object",
        "primary_label_object",
        "chosen_contact_path",
        "topview_contact_path",
        "pair_mid_path",
        "qwen_review_board_path",
        "qwen_review_board_saved",
        "ai_best_ratio",
        "ai_model_best_ratio_raw",
        "ai_object_visible_score_0_3",
        "ai_scene_clarity_score_0_3",
        "ai_topview_useful_score_0_3",
        "ai_keep_for_next_stage",
        "ai_candidate_quality",
        "ai_confidence",
        "ai_ratio_scores_json",
        "ai_reason",
        "ai_raw_text",
        "ai_status",
    ]

    for key in preferred_order:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_path(value: str, project_root: Path) -> Path:
    value = str(value or "").strip()
    if not value:
        return Path("")
    p = Path(value)
    return p if p.is_absolute() else (project_root / p).resolve()


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        return json.loads(candidate)

    raise ValueError("No valid JSON object found in model output")


def normalize_ratio(value: Any) -> str:
    try:
        x = float(value)
        ratio = f"{x:.2f}"
    except Exception:
        ratio = str(value or "").strip()

    if ratio == "0.3":
        ratio = "0.30"
    if ratio == "0.6":
        ratio = "0.60"
    if ratio == "0.9":
        ratio = "0.90"

    return ratio


def score_to_int(value: Any) -> int:
    try:
        x = int(value)
    except Exception:
        return 0
    return max(0, min(3, x))


def quality_bonus(value: Any) -> int:
    q = str(value or "").strip().lower()

    if q == "good":
        return 3
    if q == "good_enough":
        return 2
    if q == "borderline":
        return 1
    if q == "bad":
        return 0

    return 0


def normalize_quality(value: Any) -> str:
    q = str(value or "").strip().lower()

    if q in {"good", "good_enough", "borderline", "bad"}:
        return q

    return "bad"


def choose_best_from_ratio_scores(ratio_scores: list[dict[str, Any]]) -> str:
    best_ratio = ""
    best_score = -1

    for item in ratio_scores:
        ratio = normalize_ratio(item.get("ratio", ""))
        if ratio not in VALID_RATIOS:
            continue

        obj = score_to_int(item.get("object_visible_score_0_3", 0))
        scene = score_to_int(item.get("scene_clarity_score_0_3", 0))
        top = score_to_int(item.get("topview_useful_score_0_3", 0))
        qbonus = quality_bonus(item.get("candidate_quality", ""))

        # Object visibility matters most, then top-view consistency, then scene clarity.
        total = (4 * obj) + (2 * top) + scene + qbonus

        if total > best_score:
            best_score = total
            best_ratio = ratio

    return best_ratio


def select_best_item(
    ratio_scores: list[dict[str, Any]],
    best_ratio: str,
) -> dict[str, Any]:
    for item in ratio_scores:
        if normalize_ratio(item.get("ratio", "")) == best_ratio:
            return item
    return {}


def sanitize_ratio_scores(ratio_scores: Any) -> list[dict[str, Any]]:
    if not isinstance(ratio_scores, list):
        return []

    clean: list[dict[str, Any]] = []

    for item in ratio_scores:
        if not isinstance(item, dict):
            continue

        ratio = normalize_ratio(item.get("ratio", ""))
        if ratio not in VALID_RATIOS:
            continue

        clean.append(
            {
                "ratio": ratio,
                "object_visible_score_0_3": score_to_int(
                    item.get("object_visible_score_0_3", 0)
                ),
                "scene_clarity_score_0_3": score_to_int(
                    item.get("scene_clarity_score_0_3", 0)
                ),
                "topview_useful_score_0_3": score_to_int(
                    item.get("topview_useful_score_0_3", 0)
                ),
                "candidate_quality": normalize_quality(item.get("candidate_quality", "")),
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    return clean


def build_prompt(row: dict[str, str]) -> str:
    preferred_camera = row.get("preferred_camera_rule", "")
    direction = row.get("driving_direction", "")
    approach = row.get("approach", "")
    weather = row.get("weather_tags", "")
    scene = row.get("scene_tags", "")
    label_objects_json = row.get("label_objects_json", "")
    target_object = row.get("target_object", row.get("primary_label_object", ""))
    metadata_json = row.get("metadata_json", "")

    return f"""
You are a strict visual dataset-quality reviewer for a computer-vision research pipeline.

You will receive ONE review board image.

The board has 6 rows. Each row is one candidate temporal ratio:
0.15, 0.30, 0.45, 0.60, 0.75, 0.90.

Each row contains:
- the ratio label
- the selected camera frame
- the top-view frame at the same ratio

Important metadata:
- metadata JSON file: {metadata_json}
- preferred camera from JSON driving direction: {preferred_camera}
- driving direction from JSON: {direction}
- approach from JSON: {approach}
- weather tags from JSON: {weather}
- scene tags from JSON: {scene}
- label objects from JSON: {label_objects_json}
- target object selected from JSON label objects: {target_object}

Important rules:
- The preferred camera has already been selected from JSON driving direction.
- Do NOT change the camera.
- Only evaluate which temporal ratio is best.
- Score EACH ratio independently.
- Do not give the same score to every ratio unless all six rows are truly equivalent.
- Be selective. It is allowed to mark frames as borderline or bad.

Scoring guide:

object_visible_score_0_3:
- 0 = target object not visible or impossible to identify
- 1 = weak visibility, tiny, ambiguous, or heavily occluded
- 2 = usable but not ideal
- 3 = clear target object, useful for dataset curation

scene_clarity_score_0_3:
- 0 = unusable image
- 1 = poor lighting, blur, or occlusion
- 2 = usable scene
- 3 = clear scene

topview_useful_score_0_3:
- 0 = top-view is not useful or inconsistent
- 1 = weak top-view support
- 2 = usable top-view support
- 3 = top-view clearly supports this candidate ratio

candidate_quality:
- good = strong dataset candidate
- good_enough = usable but not ideal
- borderline = uncertain or weak
- bad = not suitable

Return STRICT JSON only with exactly this structure:
{{
  "ratio_scores": [
    {{
      "ratio": "0.15",
      "object_visible_score_0_3": 0,
      "scene_clarity_score_0_3": 0,
      "topview_useful_score_0_3": 0,
      "candidate_quality": "good|good_enough|borderline|bad",
      "reason": "short reason"
    }},
    {{
      "ratio": "0.30",
      "object_visible_score_0_3": 0,
      "scene_clarity_score_0_3": 0,
      "topview_useful_score_0_3": 0,
      "candidate_quality": "good|good_enough|borderline|bad",
      "reason": "short reason"
    }},
    {{
      "ratio": "0.45",
      "object_visible_score_0_3": 0,
      "scene_clarity_score_0_3": 0,
      "topview_useful_score_0_3": 0,
      "candidate_quality": "good|good_enough|borderline|bad",
      "reason": "short reason"
    }},
    {{
      "ratio": "0.60",
      "object_visible_score_0_3": 0,
      "scene_clarity_score_0_3": 0,
      "topview_useful_score_0_3": 0,
      "candidate_quality": "good|good_enough|borderline|bad",
      "reason": "short reason"
    }},
    {{
      "ratio": "0.75",
      "object_visible_score_0_3": 0,
      "scene_clarity_score_0_3": 0,
      "topview_useful_score_0_3": 0,
      "candidate_quality": "good|good_enough|borderline|bad",
      "reason": "short reason"
    }},
    {{
      "ratio": "0.90",
      "object_visible_score_0_3": 0,
      "scene_clarity_score_0_3": 0,
      "topview_useful_score_0_3": 0,
      "candidate_quality": "good|good_enough|borderline|bad",
      "reason": "short reason"
    }}
  ],
  "model_best_ratio": "0.15 or 0.30 or 0.45 or 0.60 or 0.75 or 0.90",
  "confidence": "high|medium|low",
  "overall_reason": "one short sentence"
}}

Do not output markdown.
Do not output bullets.
Do not output extra text.
""".strip()


def review_one_row(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    row: dict[str, str],
    project_root: Path,
    max_new_tokens: int = 700,
) -> dict[str, Any]:
    board_path = resolve_path(row.get("qwen_review_board_path", ""), project_root)

    if not str(board_path):
        raise FileNotFoundError("Missing qwen_review_board_path value")

    if not board_path.exists():
        raise FileNotFoundError(f"Missing Qwen review board: {board_path}")

    if str(row.get("qwen_review_board_saved", "")).strip() not in {"", "1", "1.0", "True", "true"}:
        raise RuntimeError(
            f"qwen_review_board_saved is not true for sequence {row.get('sequence_id', '')}"
        )

    prompt = build_prompt(row)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(board_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed = extract_json_block(output_text)

    ratio_scores = sanitize_ratio_scores(parsed.get("ratio_scores", []))
    calculated_best_ratio = choose_best_from_ratio_scores(ratio_scores)
    model_best_ratio = normalize_ratio(parsed.get("model_best_ratio", ""))

    best_item = select_best_item(ratio_scores, calculated_best_ratio)

    best_quality = normalize_quality(best_item.get("candidate_quality", "bad"))
    keep = "1" if best_quality in {"good", "good_enough"} else "0"

    result = {
        "ai_best_ratio": calculated_best_ratio,
        "ai_model_best_ratio_raw": model_best_ratio,
        "ai_object_visible_score_0_3": str(
            best_item.get("object_visible_score_0_3", "")
        ),
        "ai_scene_clarity_score_0_3": str(
            best_item.get("scene_clarity_score_0_3", "")
        ),
        "ai_topview_useful_score_0_3": str(
            best_item.get("topview_useful_score_0_3", "")
        ),
        "ai_keep_for_next_stage": keep,
        "ai_candidate_quality": best_quality,
        "ai_confidence": str(parsed.get("confidence", "")).strip(),
        "ai_ratio_scores_json": json.dumps(ratio_scores, ensure_ascii=False),
        "ai_reason": str(parsed.get("overall_reason", "")).strip(),
        "ai_raw_text": output_text.strip(),
    }

    del image_inputs, video_inputs
    del inputs
    del generated_ids
    del trimmed_ids

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-review aligned Qwen review boards with Qwen2.5-VL"
    )
    parser.add_argument("--profile", type=str, default="colab_drive")
    parser.add_argument(
        "--review-sheet",
        type=Path,
        default=None,
        help="Path to qwen_review_input.csv. Must contain qwen_review_board_path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default="rule_ready_sample",
        help="Used only for default input/output paths",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Process only the first N rows after start index; use 0 for all",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Row offset for chunked processing",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows even if they already exist in the output CSV",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=700,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    review_sheet = (
        args.review_sheet.resolve()
        if args.review_sheet
        else (
            paths.outputs_dir
            / "reviews"
            / args.sample_name
            / "qwen_review_input.csv"
        ).resolve()
    )

    out_csv = (
        args.out.resolve()
        if args.out
        else (
            paths.outputs_dir
            / "reviews"
            / args.sample_name
            / "qwen_auto_review_board_v1.csv"
        ).resolve()
    )

    if not review_sheet.exists():
        raise FileNotFoundError(f"Review input CSV not found: {review_sheet}")

    rows = load_csv(review_sheet)

    if args.start_idx > 0:
        rows = rows[args.start_idx:]

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    existing_rows: list[dict[str, Any]] = []
    processed_ids: set[str] = set()

    if out_csv.exists() and not args.overwrite:
        existing_rows = load_csv(out_csv)
        for r in existing_rows:
            seq_id = str(r.get("sequence_id", "")).strip()
            status = str(r.get("ai_status", "")).strip().lower()

            # Skip already attempted rows, including errors.
            # Use --overwrite if you want to rerun everything.
            if seq_id and status:
                processed_ids.add(seq_id)

    rows_to_process = [
        r for r in rows
        if str(r.get("sequence_id", "")).strip() not in processed_ids
    ]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for practical Qwen2.5-VL-7B review.")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # The review board is one image but tall. This bounds visual tokens.
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=960 * 28 * 28,
    )

    output_rows: list[dict[str, Any]] = existing_rows.copy()
    errors = 0
    started = time.time()

    for row in tqdm(rows_to_process, desc="Qwen board auto-review", unit="seq"):
        base = dict(row)
        seq_id = str(base.get("sequence_id", "")).strip()

        try:
            result = review_one_row(
                model=model,
                processor=processor,
                row=row,
                project_root=paths.project_root,
                max_new_tokens=args.max_new_tokens,
            )
            base.update(result)
            base["ai_status"] = "ok"

        except Exception as e:
            errors += 1
            base.update(
                {
                    "ai_best_ratio": "",
                    "ai_model_best_ratio_raw": "",
                    "ai_object_visible_score_0_3": "",
                    "ai_scene_clarity_score_0_3": "",
                    "ai_topview_useful_score_0_3": "",
                    "ai_keep_for_next_stage": "",
                    "ai_candidate_quality": "",
                    "ai_confidence": "",
                    "ai_ratio_scores_json": "",
                    "ai_reason": "",
                    "ai_raw_text": "",
                    "ai_status": f"error: {type(e).__name__}: {e}",
                }
            )

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_rows.append(base)

        # Save after every row so interrupted runs can resume safely.
        write_csv(output_rows, out_csv)

        if seq_id:
            processed_ids.add(seq_id)

    elapsed = time.time() - started

    print(f"profile        : {paths.profile_name}")
    print(f"review_sheet    : {review_sheet}")
    print(f"output_csv      : {out_csv}")
    print(f"existing_rows   : {len(existing_rows)}")
    print(f"rows_attempted  : {len(rows_to_process)}")
    print(f"rows_total_out  : {len(output_rows)}")
    print(f"errors          : {errors}")
    print(f"elapsed_sec     : {elapsed:.1f}")


if __name__ == "__main__":
    main()