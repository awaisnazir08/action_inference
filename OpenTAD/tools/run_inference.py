import os
import json
import torch
import numpy as np
from mmengine.config import Config
from opentad.models import build_detector

def run_single_feature_inference(config_path, checkpoint_path, npy_path, output_json, meta):
    # Load config and model
    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load feature
    features = np.load(npy_path)
    features = torch.from_numpy(features).float().unsqueeze(0).to(device)  # Add batch dim

    # Prepare minimal meta info (adjust as needed)
    # Example meta: {'video_name': 'my_video', 'duration': 60.0, 'fps': 30, ...}
    # You must provide the correct meta fields expected by your model's forward/post-processing
    metas = [meta]

    # Prepare input dict (adjust keys as needed for your model)
    # Commonly, the key is 'feats' or similar
    data_dict = {
        "feats": features,
        "metas": metas
    }

    # Inference
    with torch.no_grad():
        results = model(
            **data_dict,
            return_loss=False,
            infer_cfg=cfg.inference,
            post_cfg=cfg.post_processing,
            ext_cls=None,
        )

    # Save results as JSON
    result_eval = dict(results=results)
    with open(output_json, "w") as out:
        json.dump(result_eval, out, indent=2)
    print(f"Saved inference results to {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file (e.g. thumos_i3d.py)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--npy", required=True, help="Path to feature .npy file")
    parser.add_argument("--output", default="result_detection.json", help="Path to output JSON file")
    parser.add_argument("--video_name", default="my_video", help="Name for the video (used in results)")
    parser.add_argument("--duration", type=float, required=True, help="Duration of the video in seconds")
    parser.add_argument("--fps", type=float, default=30, help="FPS of the video")
    # Add more meta fields as needed for your model
    args = parser.parse_args()

    meta = {
        "video_name": args.video_name,
        "duration": args.duration,
        "fps": args.fps,
        # Add more fields if your model expects them (e.g., 'snippet_stride', 'offset_frames', etc.)
    }

    run_single_feature_inference(
        args.config,
        args.checkpoint,
        args.npy,
        args.output,
        meta
    )