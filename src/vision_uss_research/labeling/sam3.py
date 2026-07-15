"""SAM3 wrappers: text-prompted concept segmentation + point/box video tracking.

One checkpoint (facebook/sam3) serves both. torch/transformers import lazily so
the rest of the package works in CPU-only environments (tests, dry runs).

Lines marked `# API line` are the transformers-version-sensitive calls; if a
Colab transformers release renames them, check https://huggingface.co/facebook/sam3.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

SAM3_MODEL_ID = "facebook/sam3"

# Text prompts per object (appearance only - the class comes from sequence metadata).
OBJECT_PROMPTS = {
    "curbstone": "curb",
    "curbstone_side": "curb",
    "speedbump": "speed bump",
    "woodenboard": "wooden plank lying on the ground",
    "hose": "hose lying on the ground",
    "step": "step edge",
    "pole": "vertical pole",
    "squarepole": "vertical pole",
    "dummychild": "child mannequin",
    "bicyclestand": "bicycle rack",
    "cone": "traffic cone",
    "bollard": "bollard",
    "bush": "bush",
    "tree": "tree trunk",
    "car": "car",
    "stonelarge": "large stone block",
    "stonemiddle": "stone block",
    "cubestandard": "cube-shaped test object",
    "ubarrier": "metal barrier",
    "fence": "fence",
}


class Sam3Engine:
    """Loads the PCS (text) model and the tracker once; provides seed
    segmentation and click/box propagation with GPU-memory hygiene."""

    def __init__(self, device: str | None = None, score_threshold: float = 0.3,
                 model_id: str = SAM3_MODEL_ID):
        import torch
        from transformers import (Sam3Model, Sam3Processor,
                                  Sam3TrackerVideoModel, Sam3TrackerVideoProcessor)

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.score_threshold = score_threshold
        self.pcs_processor = Sam3Processor.from_pretrained(model_id)
        self.pcs_model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.tracker = (Sam3TrackerVideoModel.from_pretrained(model_id)
                        .to(self.device, dtype=self.dtype).eval())
        self.tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)

    def free(self) -> None:
        """Release cached GPU memory - call between sequences or the tracker's
        per-video memory banks accumulate until OOM."""
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def segment_text(self, image_rgb: np.ndarray, text_prompt: str) -> dict:
        """One text-prompted inference -> {'masks','boxes','scores'} numpy."""
        from PIL import Image
        pil = Image.fromarray(image_rgb)
        with self.torch.inference_mode():
            inputs = self.pcs_processor(images=pil, text=text_prompt,
                                        return_tensors="pt").to(self.device)  # API line
            outputs = self.pcs_model(**inputs)
            results = self.pcs_processor.post_process_instance_segmentation(  # API line
                outputs, threshold=self.score_threshold, mask_threshold=0.5,
                target_sizes=[(pil.height, pil.width)],
            )[0]
        return {"masks": results["masks"].cpu().numpy().astype(bool),
                "boxes": results["boxes"].cpu().numpy(),
                "scores": results["scores"].cpu().numpy()}

    def _track_forward(self, frames: list, seed_idx: int,
                       prompt: dict) -> dict[int, np.ndarray]:
        """prompt: {'point': (x, y)} or {'box': (x0, y0, x1, y1)}."""
        torch = self.torch
        h, w = frames[0].height, frames[0].width
        with torch.inference_mode():
            session = self.tracker_processor.init_video_session(           # API line
                video=frames, inference_device=self.device, dtype=self.dtype)
            if prompt.get("box"):
                seed_kwargs = dict(input_boxes=[[list(map(float, prompt["box"]))]])
            else:
                seed_kwargs = dict(input_points=[[[list(prompt["point"])]]],
                                   input_labels=[[[1]]])
            self.tracker_processor.add_inputs_to_inference_session(        # API line
                session, frame_idx=seed_idx, obj_ids=1, **seed_kwargs)
            masks = {}
            for output in self.tracker.propagate_in_video_iterator(        # API line
                    session, start_frame_idx=seed_idx):
                video_masks = self.tracker_processor.post_process_masks(
                    [output.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                masks[output.frame_idx] = video_masks[0, 0].cpu().numpy().astype(bool)
        return masks

    def propagate(self, frames_dir: Path, seed_idx: int,
                  prompt: dict) -> dict[int, np.ndarray]:
        """Propagate a seed prompt across all JPEGs in frames_dir. Frames before
        the seed are labeled by re-running the forward pass on the reversed list
        (no reverse-propagation API dependency)."""
        from PIL import Image
        frames = [Image.open(p).convert("RGB")
                  for p in sorted(Path(frames_dir).glob("*.jpg"))]
        results = self._track_forward(frames, seed_idx, prompt)
        self.free()
        if seed_idx > 0:
            rev = self._track_forward(list(reversed(frames)),
                                      len(frames) - 1 - seed_idx, prompt)
            for ridx, mask in rev.items():
                results.setdefault(len(frames) - 1 - ridx, mask)
        self.free()
        return results
