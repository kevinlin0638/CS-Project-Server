import cv2
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionInpaintPipelineLegacy
from PIL import Image

from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]

faceid_embeds = []
for image in images:
    image = cv2.imread("assets/images/face.jpg")
    faces = app.get(image)
    faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
faceid_embeds = torch.cat(faceid_embeds, dim=1)


base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "models/ip-adapter-faceid-portrait_sd15.bin"
image_encoder_path = "models/image_encoder/"
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
pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)


# load ip-adapter
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

face = Image.open("assets/images/myface.jpg")
masked_image = Image.open("assets/images/suit_man.jpg")
mask = Image.open("assets/images/suit_man_mask.jpg")

images = ip_model.generate(
    # prompt=prompt,
    # negative_prompt=negative_prompt,
    faceid_embeds=faceid_embeds,
    pil_image=face,
    image=masked_image,
    mask_image=mask,
    num_samples=4, num_inference_steps=50, seed=2024, strength=0.9, width=681, height=1023
)

print('Completed')
grid = image_grid(images, 1, 4)
grid.show()
