from __future__ import annotations

import argparse
import csv
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

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove fenced code block if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Find first JSON object
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
You are reviewing contact sheets for a computer-vision dataset curation pipeline.

Context:
- preferred camera: {preferred_camera}
- driving direction: {direction}
- approach: {approach}
- weather: {weather}
- scene tags: {scene}
- primary target object: {primary_object}

Images:
1. Chosen-camera contact sheet with 6 candidate ratios: 0.15, 0.30, 0.45, 0.60, 0.75, 0.90
2. Top-view contact sheet with the same ratios
3. Mid pair image for quick sanity check

Task:
Pick the BEST candidate ratio where the target obstacle is most visible and best aligned with the top-view.
Then score:
- object_visible_score_0_3: 0,1,2,3
- scene_clarity_score_0_3: 0,1,2,3
- topview_useful_score_0_3: 0,1,2,3

Scoring guide:
- 0 = not useful
- 1 = weak
- 2 = usable
- 3 = clear

Return STRICT JSON only with exactly these keys:
{{
  "best_ratio": "0.15 or 0.30 or 0.45 or 0.60 or 0.75 or 0.90",
  "object_visible_score_0_3": 0,
  "scene_clarity_score_0_3": 0,
  "topview_useful_score_0_3": 0,
  "keep_for_next_stage": 0,
  "candidate_quality": "good|good_enough|borderline|bad",
  "confidence": "high|medium|low",
  "reason": "one short sentence"
}}

Rules:
- keep_for_next_stage = 1 only if the candidate looks usable for dataset curation.
- candidate_quality should reflect the selected best_ratio only.
- Do not output markdown, bullets, or extra text.
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

    return {
        "ai_best_ratio": normalize_ratio(parsed.get("best_ratio", "")),
        "ai_object_visible_score_0_3": str(parsed.get("object_visible_score_0_3", "")),
        "ai_scene_clarity_score_0_3": str(parsed.get("scene_clarity_score_0_3", "")),
        "ai_topview_useful_score_0_3": str(parsed.get("topview_useful_score_0_3", "")),
        "ai_keep_for_next_stage": normalize_keep(parsed.get("keep_for_next_stage", "")),
        "ai_candidate_quality": str(parsed.get("candidate_quality", "")).strip(),
        "ai_confidence": str(parsed.get("confidence", "")).strip(),
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for practical Qwen2.5-VL-7B review in Colab.")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    # Keep image token usage controlled for contact sheets.
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=960 * 28 * 28,
    )

    output_rows: list[dict[str, Any]] = []
    errors = 0
    started = time.time()

    for idx, row in enumerate(tqdm(rows, desc="Qwen auto-review", unit="seq"), start=1):
        base = dict(row)
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
                    "ai_status": f"error: {type(e).__name__}: {e}",
                }
            )
        output_rows.append(base)

        # Write incrementally so a Colab disconnect does not lose progress.
        if idx % 5 == 0:
            write_csv(output_rows, out_csv)

    write_csv(output_rows, out_csv)

    elapsed = time.time() - started
    print(f"profile        : {paths.profile_name}")
    print(f"review_sheet    : {review_sheet}")
    print(f"output_csv      : {out_csv}")
    print(f"rows_processed  : {len(output_rows)}")
    print(f"errors          : {errors}")
    print(f"elapsed_sec     : {elapsed:.1f}")


if __name__ == "__main__":
    main()