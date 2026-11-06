#!/usr/bin/env python3
"""Standalone FGDN inference script (no BasicSR installation required).

Derains all images in INPUT_DIR with the checkpoint WEIGHTS and saves the
results to RESULT_DIR. Model hyper-parameters are read from the training
config YAML_PATH.
"""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


# ==================== Edit this block ====================
INPUT_DIR = Path("./datasets/Rain200H/test/input")
RESULT_DIR = Path("./results/Rain200H")
WEIGHTS = Path("./pretrained/Rain200H.pth")
YAML_PATH = Path("./options/Rain200H.yml")

# Path added to sys.path so that the model module can be imported.
EXTRA_PATH = Path(".")

# Model location; change both lines together when swapping models.
MODEL_MODULE = "basicsr.archs.FGDN_arch"
MODEL_CLASS = "FGDN"

PAD_FACTOR = 8
SAVE_EXT = ".png"
# ==========================================================


def load_model():
    sys.path.append(str(EXTRA_PATH.resolve()))

    module = __import__(MODEL_MODULE, fromlist=[MODEL_CLASS])
    model_cls = getattr(module, MODEL_CLASS)

    with YAML_PATH.open("r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    model_kwargs = dict(opt["network_g"])
    model_kwargs.pop("type", None)

    model = model_cls(**model_kwargs)

    checkpoint = torch.load(WEIGHTS, map_location="cpu")
    state_dict = checkpoint["params"] if "params" in checkpoint else checkpoint
    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model


def load_image(path: Path, device: torch.device) -> tuple[torch.Tensor, int, int]:
    image = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    h, w = image.shape[:2]
    return tensor, h, w


def save_image(tensor: torch.Tensor, path: Path) -> None:
    image = torch.clamp(tensor, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()

    files = sorted(list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.jpg")))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path in tqdm(files):
            x, h, w = load_image(path, device)

            H = ((h + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
            W = ((w + PAD_FACTOR - 1) // PAD_FACTOR) * PAD_FACTOR
            x = F.pad(x, (0, W - w, 0, H - h), mode="reflect")

            y = model(x)
            y = y[:, :, :h, :w]

            save_path = RESULT_DIR / f"{path.stem}{SAVE_EXT}"
            save_image(y, save_path)

    print(f"Done. Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
