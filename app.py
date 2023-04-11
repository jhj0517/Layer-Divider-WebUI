import gradio as gr

from modules import sam
from modules.ui_utils import *
from modules.html_constants import *


class App:
    def __init__(self):
        self.app = gr.Blocks(css=CSS)
        self.sam = sam.SamInference()

    def launch(self):
        with self.app:
            with gr.Row():
                gr.Markdown(PROJECT_NAME, elem_id="md_project")
            with gr.Row().style(equal_height=True):  # bug https://github.com/gradio-app/gradio/issues/3202
                with gr.Column(scale=5):
                    img_input = gr.Image(label="Input image here")
                with gr.Column(scale=5):
                    # Tuable Params
                    nb_points_per_side = gr.Number(label="points_per_side", value=32)
                    sld_pred_iou_thresh = gr.Slider(label="pred_iou_thresh", value=0.88, minimum=0, maximum=1)
                    sld_stability_score_thresh = gr.Slider(label="stability_score_thresh", value=0.95, minimum=0,
                                                           maximum=1)
                    nb_crop_n_layers = gr.Number(label="crop_n_layers", value=0)
                    nb_crop_n_points_downscale_factor = gr.Number(label="crop_n_points_downscale_factor", value=1)
                    nb_min_mask_region_area = gr.Number(label="min_mask_region_area", value=0)
                    html_param_explain = gr.HTML(PARAMS_EXPLANATION, elem_id="html_param_explain")

            with gr.Row():
                btn_generate = gr.Button("GENERATE", variant="primary")
            with gr.Row():
                gallery_output = gr.Gallery(label="Output will be shown here", show_label=True).style(columns=5,
                                                                                                      height="auto")
                btn_open_folder = gr.Button("📁\n(PSD)").style(full_width=False)

            params = [nb_points_per_side, sld_pred_iou_thresh, sld_stability_score_thresh, nb_crop_n_layers,
                      nb_crop_n_points_downscale_factor, nb_min_mask_region_area]
            btn_generate.click(fn=self.sam.generate_mask_app, inputs=[img_input] + params, outputs=gallery_output)
            btn_open_folder.click(fn=lambda: open_folder("outputs\psd"), inputs=None, outputs=None)

        self.app.queue(api_open=False).launch()


if __name__ == "__main__":
    app = App()
    app.launch()
