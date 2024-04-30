import os

import cv2
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image

from detect import getAgeGender
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# generate image


def generate_image(wearing, shape, file_list, uid):
    torch.cuda.empty_cache()
    gender, age = getAgeGender('uploaded/' + file_list[0]['name'])
    is_man = gender == 'Male'
    if is_man:
        prompt = "A {shape} man {wearing} stands empty-handed in a brightly lit professional photography studio, captured in a high-quality, sharp, full-body portrait with correct anatomy, avoiding monochrome, low-res, or blurry elements."
    else:
        prompt = "A {shape} woman {wearing} stands empty-handed in a brightly lit professional photography studio, captured in a high-quality, sharp, full-body portrait with correct anatomy, avoiding monochrome, low-res, or blurry elements."

    negative_prompt = "topless, wearing hat or cap, bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs, cartoon, cg, 3d, unreal, animate"

    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    ip_ckpt = "models/ip-adapter-faceid-portrait_sd15.bin"
    device = "cuda"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    # load ip-adapter
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces_url = [('uploaded/' + i['name']) for i in file_list]
    faceid_embeds = []
    for url in faces_url:
        image = cv2.imread(url)
        faces = app.get(image)
        faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
    faceid_embeds = torch.cat(faceid_embeds, dim=1)

    prompt = prompt.format(shape=shape, wearing=wearing)
    images = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=faceid_embeds,
        num_samples=4, width=512, height=1024, num_inference_steps=96, seed=2023
    )

    images[0].save(os.path.join("./public", str(uid) + "_output_0.png"))
    images[1].save(os.path.join("./public", str(uid) + "_output_1.png"))
    images[2].save(os.path.join("./public", str(uid) + "_output_2.png"))
    images[3].save(os.path.join("./public", str(uid) + "_output_3.png"))

    del pipe, ip_model
