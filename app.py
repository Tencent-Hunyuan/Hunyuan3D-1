# Open Source Model Licensed under the Apache License Version 2.0 
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited 
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import warnings
import argparse
import gradio as gr
from glob import glob
import shutil
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import pandas as pd

# import spaces
from infer import seed_everything, save_gif
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from third_party.check import check_bake_available

try:
    from third_party.mesh_baker import MeshBaker
    assert check_bake_available()
    BAKE_AVAILEBLE = True
except Exception as err:
    print(err)
    print("import baking related fail, run without baking")
    BAKE_AVAILEBLE = False

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--use_lite", default=False, action="store_true")
parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
parser.add_argument("--save_memory", default=False)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument(
    "--server_name", default="127.0.0.1", type=str,
    help="Address to bind the demo to. Use 0.0.0.0 to explicitly allow network access.",
)
args = parser.parse_args()


################################################################
# initial setting
################################################################


CONST_HEADER = '''
<h2><a href='https://github.com/tencent/Hunyuan3D-1' target='_blank'><b>Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation</b></a></h2>
⭐️Technical report: <a href='https://arxiv.org/pdf/2411.02293' target='_blank'>ArXiv</a>. ⭐️Code: <a href='https://github.com/tencent/Hunyuan3D-1' target='_blank'>GitHub</a>.
'''

CONST_NOTE = '''
❗️❗️❗️Usage❗️❗️❗️<br>

Limited by format, the model can only export *.obj mesh with vertex colors. The "face" mod can only work on *.glb.<br>
Please click "Do Rendering" to export a GIF.<br>
You can click "Do Baking" to bake multi-view imgaes onto the shape.<br>

If the results aren't satisfactory, please try a different radnom seed (default is 0).
'''

################################################################
# prepare text examples and image examples
################################################################

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./demos/example_*.png'))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list  = list()
    for line in open('./demos/example_list.txt'):
        txt_list.append(line.strip())
    return txt_list

example_is = get_example_img_list()
example_ts = get_example_txt_list()

################################################################
# initial models
################################################################

worker_xbg = Removebg()
print(f"loading {args.text2image_path}")
worker_t2i = Text2Image(
    pretrain = args.text2image_path, 
    device = args.device, 
    save_memory = args.save_memory
)
worker_i2v = Image2Views(
    use_lite = args.use_lite, 
    device = args.device,
    save_memory = args.save_memory
)
worker_v23 = Views2Mesh(
    args.mv23d_cfg_path, 
    args.mv23d_ckt_path, 
    use_lite = args.use_lite, 
    device = args.device,
    save_memory = args.save_memory
)
worker_gif = GifRenderer(args.device)

if BAKE_AVAILEBLE:
    worker_baker = MeshBaker()


### functional modules    

def gen_save_folder(max_size=30):
    os.makedirs('./outputs/app_output', exist_ok=True)
    exists = set(int(_) for _ in os.listdir('./outputs/app_output') if not _.startswith("."))
    cur_id = min(set(range(max_size)) - exists) if len(exists)<max_size else -1
    if os.path.exists(f"./outputs/app_output/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"./outputs/app_output/{(cur_id + 1) % max_size}")
        print(f"remove ./outputs/app_output/{(cur_id + 1) % max_size} success !!!")
    save_folder = f'./outputs/app_output/{max(0, cur_id)}'
    os.makedirs(save_folder, exist_ok=True)
    print(f"mkdir {save_folder} suceess !!!")
    return save_folder


# @spaces.GPU(duration=150)
def gen_pipe(text, image=None, do_removebg=True, sseed=0, sstep=25, SSEED=0, SSTEP=50, color='face', 
        bake=False, render=True, max_faces=12000, force=False, front='auto', others=[180], align_times=3):
    save_folder = gen_save_folder()
    image_gen = image is not None
    if not image_gen:
        image = worker_t2i(text, sseed, sstep)
        image.save(save_folder + '/img.png')
    img_nobg = worker_xbg(image, force=do_removebg if image_gen else True)
    img_nobg.save(save_folder + '/img_nobg.png')
    yield img_nobg, None, None, None, None, None
    res_img, pils = worker_i2v(img_nobg, seed=SSEED, steps=SSTEP)
    save_gif(pils,  save_folder + '/views.gif')
    views_img, cond_img = res_img[0], res_img[1]
    img_array = np.asarray(views_img, dtype=np.uint8)
    show_img = rearrange(img_array, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_img = rearrange(show_img[worker_i2v.order, ...], '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_img = Image.fromarray(show_img)
    yield img_nobg, show_img, None, None, None, None
    do_texture_mapping = color == 'face'
    worker_v23(
        views_img, cond_img, seed = SSEED,
        save_folder = save_folder,
        target_face_count = max_faces,
        do_texture_mapping = do_texture_mapping
    )
    glb_v23 = save_folder + '/mesh.glb' if do_texture_mapping else None
    obj_v23 = save_folder + '/mesh.obj'
    obj_v23 = save_folder + '/mesh_vertex_colors.obj'
    yield img_nobg, show_img, obj_v23, glb_v23, None, None
    glb_dst = None
    if do_texture_mapping and bake:
        obj_dst = worker_baker(save_folder, force, front, others, align_times)
        glb_dst = obj_dst.replace(".obj", ".glb")
    yield img_nobg, show_img, obj_v23, glb_v23, glb_dst, None
    if do_texture_mapping and render: 
        baked_obj_list = sorted(glob(save_folder + '/view_*/bake/mesh.obj'))
        obj_dst = baked_obj_list[-1] if len(baked_obj_list)>=1 else save_folder+'/mesh.obj'
        assert os.path.exists(obj_dst), f"{obj_dst} file not found"
        gif_dst = obj_dst.replace(".obj", ".gif")
        worker_gif(obj_dst, gif_dst_path=gif_dst)
        yield img_nobg, show_img, obj_v23, glb_v23, glb_dst, gif_dst
        
def check_image_available(image):
    if image is None:
        return "Please upload image", gr.update()
    elif not hasattr(image, 'mode'):
        return "Not support, please upload other image", gr.update()
    elif image.mode == "RGBA":
        data = np.array(image)
        alpha_channel = data[:, :, 3]
        unique_alpha_values = np.unique(alpha_channel)
        if len(unique_alpha_values) == 1:
            msg = "The alpha channel is missing or invalid. The background removal option is selected for you."
            return msg, gr.update(value=True, interactive=False)
        else:
            msg = "The image has four channels, and you can choose to remove the background or not."
            return msg, gr.update(value=False, interactive=True)
    elif image.mode == "RGB":
        msg = "The alpha channel is missing or invalid. The background removal option is selected for you."
        return msg, gr.update(value=True, interactive=False)
    else:
        raise Exception("Image Error")
    

def update_mode(mode):
    color_change = {
        'Vertex color': gr.update(value='vertex'),
        'Face color': gr.update(value='face'),
        'Baking': gr.update(value='face')
    }[mode]
    bake_change = {
        'Vertex color': gr.update(value=False, interactive=False, visible=False),
        'Face color': gr.update(value=False),
        'Baking': gr.update(value=BAKE_AVAILEBLE)
    }[mode]
    face_change = {
        'Vertex color': gr.update(value=120000, maximum=300000),
        'Face color': gr.update(value=60000, maximum=300000),
        'Baking': gr.update(value=10000, maximum=60000)
    }[mode]
    render_change = {
        'Vertex color': gr.update(value=False, interactive=False, visible=False),
        'Face color': gr.update(value=True),
        'Baking': gr.update(value=True)
    }[mode]
    return color_change, bake_change, face_change, render_change
    
# ===============================================================
# gradio display
# ===============================================================

with gr.Blocks() as demo:
    gr.Markdown(CONST_HEADER)
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            with gr.Tab("Text to 3D"):
                with gr.Column():
                    text = gr.TextArea('一只黑白相间的熊猫在白色背景上居中坐着，呈现出卡通风格和可爱氛围。', 
                                       lines=3, max_lines=20, label='Input text (within 70 words)')
                with gr.Row():
                    gr.Examples(examples=example_ts, inputs=[text], label="Text examples", examples_per_page=10)
                with gr.Row():
                    textgen_submit = gr.Button("Generate", variant="primary") 

            with gr.Tab("Image to 3D"):
                with gr.Row():
                    input_image = gr.Image(label="Input image", width=256, height=256, type="pil",
                                           image_mode="RGBA", sources="upload", interactive=True)
                with gr.Row():
                    alert_message = gr.Markdown("")  # for warning 
                with gr.Row():
                    gr.Examples(examples=example_is,  inputs=[input_image], 
                        label="Img examples", examples_per_page=10)
                with gr.Row():
                    removebg = gr.Checkbox(
                        label="Remove Background", 
                        value=True, 
                        interactive=True
                    )
                    imggen_submit = gr.Button("Generate", variant="primary")

            mode = gr.Radio(
                choices=['Vertex color', 'Face color', 'Baking'], 
                label="Texture mode",
                value='Baking',
                interactive=True
            )

            with gr.Accordion("Custom settings", open=False):
                color = gr.Radio(choices=["vertex", "face"], label="Color", value="face")
                with gr.Row():

                    render = gr.Checkbox(
                        label="Do Rendering", 
                        value=True, 
                        interactive=True
                    )
                    bake = gr.Checkbox(
                        label="Do Baking", 
                        value=True if BAKE_AVAILEBLE else False, 
                        interactive=True if BAKE_AVAILEBLE else False
                    )

                with gr.Row():
                    seed = gr.Number(value=0, label="T2I seed", precision=0, interactive=True)
                    SEED = gr.Number(value=0, label="Gen seed", precision=0, interactive=True)

                step = gr.Slider(
                    value=25,
                    minimum=15,
                    maximum=50,
                    step=1,
                    label="T2I steps",
                    interactive=True
                )
                STEP = gr.Slider(
                    value=50,
                    minimum=20,
                    maximum=80,
                    step=1,
                    label="Gen steps",
                    interactive=True
                )
                max_faces = gr.Slider(
                    value=10000,
                    minimum=2000,
                    maximum=60000,
                    step=1000,
                    label="Face number limit",
                    interactive=True
                )

                with gr.Accordion("Baking Options", open=False):
                    force_bake = gr.Checkbox(
                            label="Force (Ignore the degree of matching)", 
                            value=False, 
                            interactive=True
                        )
                    front_baking = gr.Radio(
                        choices=['input image', 'multi-view front view', 'auto'], 
                        label="Front view baking",
                        value='auto',
                        interactive=True,
                        visible=True
                    )
                    other_views = gr.CheckboxGroup(
                        choices=['60°', '120°', '180°', '240°', '300°'], 
                        label="Other views baking",
                        value=['180°'],
                        interactive=True,
                        visible=True
                    )
                    align_times =gr.Slider(
                        value=1,
                        minimum=1,
                        maximum=5,
                        step=1,
                        label="Number of alignment attempts per view",
                        interactive=True
                    )

            input_image.change(
                fn=check_image_available, 
                inputs=input_image, 
                outputs=[alert_message, removebg]
            )

            mode.change(
                fn=update_mode,
                inputs=mode, 
                outputs=[color, bake, max_faces, render]
            )

            gr.Markdown(CONST_NOTE)

        ###### Output region

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=2):
                    rembg_image = gr.Image(
                        label="Image without background", 
                        type="pil",
                        image_mode="RGBA", 
                        interactive=False
                    )
                with gr.Column(scale=3):
                    result_image = gr.Image(
                        label="Multi-view images", 
                        type="pil", 
                        interactive=False
                    )

            result_3dobj = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="OBJ vertex color",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False
            )

            result_3dglb_texture = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="GLB face color",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False)

            result_3dglb_baked = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="GLB baking",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False)

            result_gif = gr.Image(label="GIF", interactive=False)

            with gr.Row():
                gr.Markdown(
                    "Due to Gradio limitations, OBJ files are displayed with vertex shading only, "
                    "while GLB files can be viewed with face color. <br>For the best experience, "
                    "we recommend downloading the GLB files and opening them with 3D software "
                    "like Blender or MeshLab."
                )

    #===============================================================
    # gradio running code
    #===============================================================

    none = gr.State(None)

    textgen_submit.click(
        fn=gen_pipe,
        inputs=[text, none, removebg, seed, step, SEED, STEP, color, bake, render, max_faces, force_bake,
                front_baking, other_views, align_times],
        outputs=[rembg_image, result_image, result_3dobj, result_3dglb_texture, result_3dglb_baked, result_gif],
    )

    imggen_submit.click(
        fn=gen_pipe,
        inputs=[none, input_image, removebg, seed, step, SEED, STEP, color, bake, render, max_faces, force_bake,
                front_baking, other_views, align_times],
        outputs=[rembg_image, result_image, result_3dobj, result_3dglb_texture, result_3dglb_baked, result_gif],
    )

    demo.queue(max_size=1)
    demo.launch(server_name=args.server_name, server_port=8080)
    # demo.launch()
