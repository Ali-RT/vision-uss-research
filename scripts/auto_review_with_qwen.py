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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_path(value: str, project_root: Path) -> Path:
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


def normalize_keep(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "keep"}:
        return "1"
    if s in {"0", "false", "no", "drop"}:
        return "0"
    return ""


def normalize_ratio(value: Any) -> str:
    try:
        x = float(value)
        return f"{x:.2f}"
    except Exception:
        return str(value).strip()


def build_prompt(row: dict[str, str]) -> str:
    preferred_camera = row.get("preferred_camera_rule", "")
    direction = row.get("driving_direction", "")
    approach = row.get("approach", "")
    weather = row.get("weather_tags", "")
    primary_object = row.get("primary_label_object", "")
    scene = row.get("scene_tags", "")

    return f"""
You are a strict visual dataset-quality reviewer for a computer-vision research pipeline.

You are reviewing contact sheets for near-field obstacle understanding.

Context:
- preferred camera: {preferred_camera}
- driving direction: {direction}
- approach: {approach}
- weather: {weather}
- scene tags: {scene}
- target object: {primary_object}

Images:
1. Chosen-camera contact sheet with 6 candidate ratios:
   0.15, 0.30, 0.45, 0.60, 0.75, 0.90
2. Top-view contact sheet with the same 6 candidate ratios
3. Mid pair image for sanity check only

Your task:
Choose exactly ONE best ratio. Do not default to the middle frame unless it is truly best.

Evaluate the six ratios comparatively:
- Which ratio shows the target object most clearly?
- Which ratio has the least occlusion/blur?
- Which ratio aligns best with the top-view context?
- Which ratio would be most useful as a dataset example?

Scoring:
object_visible_score_0_3:
- 0 = target object not visible or impossible to identify
- 1 = weak visibility, tiny, ambiguous, or heavily occluded
- 2 = usable but not ideal
- 3 = clear target object, good dataset example

scene_clarity_score_0_3:
- 0 = unusable image
- 1 = poor lighting/blur/occlusion
- 2 = usable
- 3 = clear scene

topview_useful_score_0_3:
- 0 = top-view is not useful or inconsistent
- 1 = weak top-view support
- 2 = usable support
- 3 = clearly supports the selected ratio

candidate_quality:
- good = object is clear, scene is clear, top-view supports the frame
- good_enough = usable, but not ideal
- borderline = uncertain, weak visibility, or weak top-view alignment
- bad = not suitable for dataset curation

keep_for_next_stage:
- 1 only for good or good_enough
- 0 for borderline or bad

Important:
Be selective. It is acceptable to output borderline or bad.
Do not assign all scores as 2 unless the selected frame is genuinely only moderately usable.
If no ratio clearly shows the target object, choose the least bad ratio and set keep_for_next_stage to 0.

Return STRICT JSON only with exactly these keys:
{{
  "best_ratio": "0.15 or 0.30 or 0.45 or 0.60 or 0.75 or 0.90",
  "object_visible_score_0_3": 0,
  "scene_clarity_score_0_3": 0,
  "topview_useful_score_0_3": 0,
  "keep_for_next_stage": 0,
  "candidate_quality": "good|good_enough|borderline|bad",
  "confidence": "high|medium|low",
  "ratio_comparison": "short comparison explaining why the selected ratio is better than nearby ratios",
  "reason": "one short sentence explaining the decision"
}}

Do not output markdown, bullets, or any extra text.
""".strip()

def review_one_row(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    row: dict[str, str],
    project_root: Path,
    max_new_tokens: int = 220,
) -> dict[str, Any]:
    chosen_contact_path = resolve_path(row["chosen_contact_path"], project_root)
    topview_contact_path = resolve_path(row["topview_contact_path"], project_root)
    pair_mid_path = resolve_path(row["pair_mid_path"], project_root)

    if not chosen_contact_path.exists():
        raise FileNotFoundError(f"Missing chosen contact sheet: {chosen_contact_path}")
    if not topview_contact_path.exists():
        raise FileNotFoundError(f"Missing topview contact sheet: {topview_contact_path}")
    if not pair_mid_path.exists():
        raise FileNotFoundError(f"Missing pair mid image: {pair_mid_path}")

    prompt = build_prompt(row)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(chosen_contact_path)},
                {"type": "image", "image": str(topview_contact_path)},
                {"type": "image", "image": str(pair_mid_path)},
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
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

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

    # Explicit cleanup before returning
    del image_inputs, video_inputs
    del inputs
    del generated_ids
    del trimmed_ids

    return {
        "ai_best_ratio": normalize_ratio(parsed.get("best_ratio", "")),
        "ai_object_visible_score_0_3": str(parsed.get("object_visible_score_0_3", "")),
        "ai_scene_clarity_score_0_3": str(parsed.get("scene_clarity_score_0_3", "")),
        "ai_topview_useful_score_0_3": str(parsed.get("topview_useful_score_0_3", "")),
        "ai_keep_for_next_stage": normalize_keep(parsed.get("keep_for_next_stage", "")),
        "ai_candidate_quality": str(parsed.get("candidate_quality", "")).strip(),
        "ai_confidence": str(parsed.get("confidence", "")).strip(),
        "ai_ratio_comparison": str(parsed.get("ratio_comparison", "")).strip(),
        "ai_reason": str(parsed.get("reason", "")).strip(),
        "ai_raw_text": output_text.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-review contact sheets with Qwen2.5-VL")
    parser.add_argument("--profile", type=str, default="colab_drive")
    parser.add_argument(
        "--review-sheet",
        type=Path,
        default=None,
        help="Override path to contact_review_sheet.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Process only the first N rows; use 0 for all",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    review_sheet = (
        args.review_sheet.resolve()
        if args.review_sheet
        else (paths.outputs_dir / "reviews" / "rule_ready_sample" / "contact_review_sheet.csv").resolve()
    )
    out_csv = (
        args.out.resolve()
        if args.out
        else (paths.outputs_dir / "reviews" / "rule_ready_sample" / "qwen_auto_review.csv").resolve()
    )

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
            # Skip rows already processed, including errors, unless overwrite is requested
            if seq_id and status:
                processed_ids.add(seq_id)

    rows_to_process = [
        r for r in rows
        if str(r.get("sequence_id", "")).strip() not in processed_ids
    ]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for practical Qwen2.5-VL-7B review in Colab.")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=960 * 28 * 28,
    )

    output_rows: list[dict[str, Any]] = existing_rows.copy()
    errors = 0
    started = time.time()

    for row in tqdm(rows_to_process, desc="Qwen auto-review", unit="seq"):
        base = dict(row)
        seq_id = str(base.get("sequence_id", "")).strip()

        try:
            result = review_one_row(model, processor, row, project_root=paths.project_root)
            base.update(result)
            base["ai_status"] = "ok"
        except Exception as e:
            errors += 1
            base.update(
                {
                    "ai_best_ratio": "",
                    "ai_object_visible_score_0_3": "",
                    "ai_scene_clarity_score_0_3": "",
                    "ai_topview_useful_score_0_3": "",
                    "ai_keep_for_next_stage": "",
                    "ai_candidate_quality": "",
                    "ai_confidence": "",
                    "ai_reason": "",
                    "ai_raw_text": "",
                    "ai_ratio_comparison": "",
                    "ai_status": f"error: {type(e).__name__}: {e}",
                }
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_rows.append(base)

        # Save after EVERY row so interrupted runs can resume safely
        write_csv(output_rows, out_csv)

        # Also mark as processed in-memory immediately
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