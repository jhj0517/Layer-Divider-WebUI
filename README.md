# Layer-Divider-WebUI
A Gradio-based browser interface for [SAM](https://github.com/facebookresearch/segment-anything). You can use it as an tool for Layer Divider

![Layer-Divider-WebUI](https://raw.githubusercontent.com/jhj0517/Layer-Divider-WebUI/master/screenshot.png?token=GHSAT0AAAAAAB7Z7LNTY3GG4N2T5DX2VYIMZBRGLZA)

# Installation and Running
## Prerequisite
you need to have `git`.

Please follow the links below to install the necessary software:
- git : [https://git-scm.com/downloads](https://git-scm.com/downloads)

make sure to add the `git` to your system PATH.

## Automatic Installation
If you have satisfied the prerequisite above, you are now ready to start WebUI.

1. Download and unzip this repository. ( or `git clone https://github.com/jhj0517/Layer-Divider-WebUI.git` )
2. Run `Install.bat` from Windows Explorer as a regular, non-administrator user.
3. After installation, run the `start-webui.bat`. 
4. Open your web browser and go to `http://localhost:7860`

( If you're running another Web-UI, it will be hosted on a different port , such as `localhost:7861`, `localhost:7862`, and so on )

## Explanation of Parameters

| Parameter                      | Description                                                                                                                                                                                                                                                                              |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| points_per_side                | The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.                                                                                                            |
| pred_iou_thresh                | A filtering threshold in [0,1], using the model's predicted mask quality.                                                                                                                                                                                                               |
| stability_score_thresh         | A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.                                                                                                                                             |
| crops_n_layers                 | If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.                                                                                                                                |
| crop_n_points_downscale_factor | The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.                                                                                                                                                                                 |
| min_mask_region_area           | If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.                                                                                                                                  |


