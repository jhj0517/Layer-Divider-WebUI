import torch

AVAILABLE_MODELS = {
    "ViT-H SAM model": ["sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"],
    "ViT-L SAM model": ["sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"],
    "ViT-B SAM model": ["sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"],
}


def download_sam_model_url():
    torch.hub.download_url_to_file(AVAILABLE_MODELS["ViT-H SAM model"][1],
                                   f'models/{AVAILABLE_MODELS["ViT-H SAM model"][0]}')
