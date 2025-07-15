import os
import sys
import time
import yaml

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger


def main():
    # Load config from yaml/config.yaml
    with open(os.path.join(path, "..", "yaml", "config.yaml"), "r") as f:
        yaml_cfg = yaml.safe_load(f)
    args = yaml_cfg["action_former_args"]

    # load config
    cfg = Config.fromfile(args["config"])
    if args.get("cfg_options") is not None:
        cfg.merge_from_dict(args["cfg_options"])

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set random seed, create work_dir
    set_seed(args["seed"])
    create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    time_start = time.time()
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )
    print(f"Time taken to load data {time.time() - time_start}")

    # build model
    model = build_detector(cfg.model)
    model = model.to(device)

    def remove_module_prefix(state_dict):
        # Remove 'module.' prefix from keys if present
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

    if cfg.inference.load_from_raw_predictions:  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:  # load checkpoint: args -> config -> best
        if args["checkpoint"] != "none":
            checkpoint_path = args["checkpoint"]
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
        logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

        # Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        state_dict = checkpoint["state_dict_ema"] if use_ema else checkpoint["state_dict"]
        state_dict = remove_module_prefix(state_dict)
        # try:
        #     model.load_state_dict(state_dict)
        # except RuntimeError as e:
        #     logger.warning(f"Strict loading failed: {e}\nTrying with strict=False...")
        #     model.load_state_dict(state_dict, strict=False)
        #     logger.warning("Model loaded with strict=False. Some keys may be missing or unexpected. Please verify model compatibility.")
        model.load_state_dict(state_dict, strict=True)

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    time_start = time.time()
    eval_one_epoch(
        test_loader,
        model,
        cfg,
        model_ema=None,  # since we have loaded the ema model above
        use_amp=use_amp,
        device=device,  # pass device
    )
    logger.info("Testing Over...\n")
    print(f"Time taken: {time.time() - time_start}")

if __name__ == "__main__":
    main()
