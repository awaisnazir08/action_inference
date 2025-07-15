import argparse
import json
import numpy as np
import torch
import sys
import os

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)


from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys if present."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_model(config_path, checkpoint_path, device="cuda:0"):
    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_ema = getattr(cfg.solver, "ema", False)
    # Try EMA first, fallback to standard
    if use_ema and "state_dict_ema" in checkpoint and checkpoint["state_dict_ema"]:
        try:
            state_dict = checkpoint["state_dict_ema"]
            model.load_state_dict(strip_module_prefix(state_dict))
        except Exception as e:
            print("EMA state dict failed, trying standard state dict:", e)
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(strip_module_prefix(state_dict))
    else:
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(strip_module_prefix(state_dict))
    model = model.to(device)
    model.eval()
    return model, cfg


def recursive_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: recursive_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_python(v) for v in obj]
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="ActionFormer Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--features", type=str, required=True, help="Path to feature .npy file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    args = parser.parse_args()

    model, cfg = load_model(args.config, args.checkpoint, args.device)

    # Build the dataset from config
    dataset = build_dataset(cfg.dataset.test)

    # Find the index for your video (e.g., 'vid_767')
    video_idx = next(i for i, item in enumerate(dataset.data_list) if item[0] == 'vid_767')

    # Get the sample
    sample = dataset[video_idx]
    feats = sample['feats'].unsqueeze(0).to(args.device)  # [1, C, T]
    masks = sample['masks'].unsqueeze(0).to(args.device)  # [1, T]
    metas = [sample['metas']]  # already contains all required fields

    post_cfg = cfg.post_processing
    post_cfg.sliding_window = False  # Ensure this is set for single-video inference
    # Load class map from config (as in dataset config)
    class_map_path = cfg.dataset.test.get("class_map", None)
    if class_map_path is not None:
        with open(class_map_path, "r", encoding="utf8") as f:
            class_map = [line.strip() for line in f.readlines()]
    else:
        class_map = None
    ext_cls = class_map

    # Use the model as a callable, just like in test.py
    results = model(
        feats, masks,
        metas=metas,
        return_loss=False,
        infer_cfg=cfg.inference,
        post_cfg=post_cfg,
        ext_cls=ext_cls
    )
    final_result = {"results": results}
    print(json.dumps(recursive_to_python(final_result), indent=2))


if __name__ == "__main__":
    main()
