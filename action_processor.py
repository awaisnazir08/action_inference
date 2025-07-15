import os
import sys
import time
import yaml
import json
import torch
import glob
import pprint
import numpy as np
from omegaconf import OmegaConf
from mmengine.config import Config
from OpenTAD.opentad.models import build_detector
from OpenTAD.opentad.datasets import build_dataset, build_dataloader
from OpenTAD.opentad.utils import update_workdir, set_seed, create_folder, setup_logger
from video_utils import VideoInfo, get_video_frames_batch_generator
from video_features.models.i3d.extract_i3d import ExtractI3D
from video_features.utils.utils import build_cfg_path
from helper import remove_module_prefix, get_video_name, check_path, create_path

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

class ActionProcessor:
    def __init__(self, config):
        self.action_former_model_path = config['model_checkpoint_path']
        self.config_file_path = config['actionformer_config_file_path']
        self.seed = config.get('seed', 42)
        self.cfg_options = config.get('cfg_options', {})
        self.feature_args = config['video_features_extraction']
        self.feature_type = self.feature_args['feature_type']
        self.features_output_dir = config['videos_features_directory']

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.feature_extractor = self._load_feature_extractor()
        self.videos_metadata_directory = config['videos_metadata_directory']

        # Load OpenTAD config
        self.cfg = self._load_opentad_config()

        # Setup logger
        self.logger = self._setup_logger()

        # Override the data paths in the config file
        self._override_config_datafile_paths(config)

        # Load post-processing config
        self._override_post_processing_config(config)

        # Set random seed
        set_seed(self.seed)


        # Build and load model
        self.action_former_model = self._load_model()

        # AMP
        self.use_amp = getattr(self.cfg.solver, "amp", False)
        if self.use_amp:
            self.logger.info("Using Automatic Mixed Precision...")

    def _load_opentad_config(self):
        cfg = Config.fromfile(self.config_file_path)
        if self.cfg_options:
            cfg.merge_from_dict(self.cfg_options)
        return cfg

    def _override_config_datafile_paths(self, config):
        self.cfg.dataset.test.class_map = config['class_map']
        self.cfg.dataset.test.data_path = self.features_output_dir
        self.cfg.dataset.test.block_list = config['block_list']
        print(f"The config file's data paths have been overwritten..!!")

    def _setup_logger(self):
        logger = setup_logger("ActionProcessor", None)
        logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
        logger.info(f"Config: \n{self.cfg.pretty_text}")
        return logger

    def _load_model(self):
        model = build_detector(self.cfg.model)
        model = model.to(self.device)
        model.eval()
        checkpoint = torch.load(self.action_former_model_path, map_location=self.device)
        self.logger.info(f"Checkpoint is epoch {checkpoint['epoch']}.")
        use_ema = getattr(self.cfg.solver, "ema", False)
        state_dict = checkpoint["state_dict_ema"] if use_ema else checkpoint["state_dict"]
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def _load_feature_extractor(self):
        # Load base config
        args = OmegaConf.load(build_cfg_path(self.feature_type))
        # Override with values from config.yaml
        for k, v in self.feature_args.items():
            setattr(args, k, v)
        args.device = str(self.device)
        args.output_path = os.path.join(self.features_output_dir)
        return ExtractI3D(args)

    def _override_post_processing_config(self, config):
        self.post_processing = config.get('post_processing', None)
        if self.post_processing is not None:
            self.logger.info(f"Loaded post_processing config from YAML: {self.post_processing}")
            # Overwrite self.cfg.post_processing with YAML values
            self.cfg.post_processing = self.post_processing
        else:
            self.logger.warning("No post_processing config found in YAML. Using defaults from model config.")

    def _run_feature_extractor(self, video_path):
        return self.feature_extractor.extract(video_path)

    def _combine_and_save_features(self, feature_dict, npy_path):
        rgb_features = feature_dict['rgb']
        flow_features = feature_dict['flow']
        if rgb_features.shape[0] != flow_features.shape[0]:
            raise ValueError("Mismatch in temporal length between RGB and Flow features.")
        combined_features = np.concatenate((rgb_features, flow_features), axis=1)
        np.save(npy_path, combined_features)
        self.logger.info(f"The features have been saved successfully in the path: {npy_path}")

    def _extract_features(self, video_path, video_name):
        npy_path = create_path(self.features_output_dir, video_name, 'npy')

        if check_path(npy_path):
            print(f"The features already exist in the path: {npy_path}")
            return npy_path
        
        feature_dict = self._run_feature_extractor(video_path)

        if feature_dict and 'rgb' in feature_dict and 'flow' in feature_dict:
            self._combine_and_save_features(feature_dict, npy_path)
            return npy_path
        else:
            raise RuntimeError("Feature extraction failed or did not return 'rgb' and 'flow'.")
    
    def _create_video_metadata(self, video_name, video_info):

        return {
            "database": {
                video_name: {
                    "subset": 'validation',
                    "duration": video_info.duration,
                    "frame": video_info.total_frames,
                    "annotations": [],
                }
            }
        }

    def _save_video_metadata(self, video_metadata, video_metadata_path):
        if check_path(video_metadata_path):
            print(f"The video metadata already exist in the path: {video_metadata_path}")
            return
        with open(video_metadata_path, "w") as f:
            json.dump(video_metadata, f, indent=2)
    
    def _build_dataset_config_for_video(self, video_metadata_path):
        dataset_cfg = self.cfg.dataset.test.copy()
        dataset_cfg["ann_file"] = video_metadata_path
        

        # # If class_map or block_list are relative, resolve them
        # if "class_map" in dataset_cfg and not os.path.isabs(dataset_cfg["class_map"]):
        #     dataset_cfg["class_map"] = os.path.join(os.path.dirname(self.config_file_path), dataset_cfg["class_map"])
        # if "block_list" in dataset_cfg and not os.path.isabs(dataset_cfg["block_list"]):
        #     dataset_cfg["block_list"] = os.path.join(os.path.dirname(self.config_file_path), dataset_cfg["block_list"])

        test_dataset = build_dataset(dataset_cfg, default_args=dict(logger=self.logger))
        test_loader = build_dataloader(
            test_dataset,
            rank=0,
            world_size=1,
            shuffle=False,
            drop_last=False,
            **self.cfg.solver.test,
        )

        return dataset_cfg, test_dataset, test_loader
    
    def perform_inference(self, inference_loader, inference_dataset):
        self.action_former_model.eval()
        result_dict = {}
        for data_dict in inference_loader:
            # Move all tensors in data_dict to the correct device
            for k, v in data_dict.items():
                if torch.is_tensor(v):
                    data_dict[k] = v.to(self.device)
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.use_amp):
                with torch.no_grad():
                    results = self.action_former_model(
                        **data_dict,
                        return_loss=False,
                        infer_cfg=self.cfg.inference,
                        post_cfg=self.cfg.post_processing,
                        ext_cls=inference_dataset.class_map,
                    )
            for k, v in results.items():
                if k in result_dict:
                    result_dict[k].extend(v)
                else:
                    result_dict[k] = v
        return result_dict

    def process(self, video_path):
        # 1. Extract video information and initialize frame generator
        video_info = VideoInfo.from_video_path(video_path)

        # 2. get the video_name
        video_name = get_video_name(video_path)

        # 3. Extract features
        features_path = self._extract_features(video_path, video_name)

        # 4. Create minimal annotation JSON for this video
        video_metadata = self._create_video_metadata(video_name, video_info)

        # 5. Create output directory if it doesn't exist
        os.makedirs(self.videos_metadata_directory, exist_ok=True)
        
        # 6. Get path for the output JSON file
        video_metadata_path = create_path(self.videos_metadata_directory, video_name, "json")
        
        # 7. Save the metadata to JSON
        self._save_video_metadata(video_metadata, video_metadata_path)
        
        # 8. Build dataset config for this video
        dataset_cfg, test_dataset, test_loader = self._build_dataset_config_for_video(video_metadata_path)

        # 6. Run inference
        self.action_former_model.eval()
        result_dict = {}
        for data_dict in test_loader:
            # Move all tensors in data_dict to the correct device
            for k, v in data_dict.items():
                if torch.is_tensor(v):
                    data_dict[k] = v.to(self.device)
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.use_amp):
                with torch.no_grad():
                    results = self.action_former_model(
                        **data_dict,
                        return_loss=False,
                        infer_cfg=self.cfg.inference,
                        post_cfg=self.cfg.post_processing,
                        ext_cls=test_dataset.class_map,
                    )
            for k, v in results.items():
                if k in result_dict:
                    result_dict[k].extend(v)
                else:
                    result_dict[k] = v
        # Clean up temp annotation file
        # os.remove(video_metadata_path)
        # os.remove()
        return result_dict


if __name__ == "__main__":
    with open("./yaml/config.yml", 'r') as config:
        config = yaml.safe_load(config)
        action_processor_config = config["action_former_args"]
    
    action = ActionProcessor(action_processor_config)
    results = action.process(r'E:\VS Code Folders\i3d\videos\vid_620.mp4')
    pprint.pprint(results)
