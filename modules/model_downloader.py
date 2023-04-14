import os
import torch

DEFAULT_MODEL_TYPE = "vit_h"

AVAILABLE_MODELS = {
    "vit_h": ["sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"],
    "vit_l": ["sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"],
    "vit_b": ["sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"],
}

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sam_model_path = os.path.join(base_dir, "models")


def download_sam_model_url(model_type):
    model_path = os.path.join(sam_model_path, AVAILABLE_MODELS[model_type][0])
    torch.hub.download_url_to_file(AVAILABLE_MODELS[model_type][1], model_path)


def is_sam_exist(model_type):
    model_path = os.path.join(sam_model_path, AVAILABLE_MODELS[model_type][0])
    if not os.path.exists(model_path):
        return False
    else:
        return True
