"""Hugging Face feature extractors (DINOv2, CLIP). Heavy imports are lazy so the rest of the
project works without torch/transformers installed."""
import numpy as np
from PIL import Image

HF_IDS = {"DINOv2": "facebook/dinov2-base", "CLIP": "openai/clip-vit-base-patch32"}
_cache = {}


def is_hf(name):
    return name in HF_IDS


def _load(name):
    if name not in _cache:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if name == "CLIP":
            model = CLIPModel.from_pretrained(HF_IDS[name]); proc = CLIPProcessor.from_pretrained(HF_IDS[name])
        else:
            model = AutoModel.from_pretrained(HF_IDS[name]); proc = AutoImageProcessor.from_pretrained(HF_IDS[name])
        _cache[name] = (model.to(device).eval(), proc, device)
    return _cache[name]


def hf_extract(name, images, bs=64):
    """images: sequence of PIL images or uint8 HxWx3 arrays -> (N, D) float32 features.
    DINOv2: [CLS token ; mean patch token] (1536-d). CLIP: image embedding (512-d)."""
    import torch
    model, proc, device = _load(name)
    out = []
    for i in range(0, len(images), bs):
        pil = [im.convert("RGB") if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im)).convert("RGB")
               for im in images[i:i + bs]]
        inputs = proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            if name == "CLIP":
                f = model.get_image_features(**inputs)
            else:
                h = model(**inputs).last_hidden_state
                f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        out.append(f.cpu().numpy())
    return np.concatenate(out)
