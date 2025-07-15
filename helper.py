import os

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from all keys in a state_dict (for DDP checkpoints).
    """
    return {k.replace("module.", ""): v for k, v in state_dict.items()} 

def get_video_name(video_path):
    """
    Extracts the video name (without extension) from the full file path.

    Args:
        video_path (str): Full path to the video file.

    Returns:
        str: Video name without extension.
    """
    base_name = os.path.basename(video_path)      # e.g. "vid_767.mp4"
    video_name = os.path.splitext(base_name)[0]   # e.g. "vid_767"
    return video_name

def check_path(path):
    return os.path.exists(path)

def create_path(base_path, name, extension):
    return os.path.join(base_path, f"{name}.{extension}")
