from diffusers import UNet2DModel, DDIMScheduler, DDIMInverseScheduler
import torch
import torch.nn.functional as F
from PIL import Image

scheduler = DDIMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", clip_sample=False, set_alpha_to_one=True)
inv_scheduler = DDIMInverseScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", clip_sample=False, set_alpha_to_one=True)

model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to('cuda')
model.requires_grad_(False)

scheduler.set_timesteps(50)
inv_scheduler.set_timesteps(50)

def latent2image(image, return_pil:bool=True):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    if not return_pil:
        return image
    image = Image.fromarray(image)
    return image

def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1 + torch.sin(alpha * theta) / torch.sin(theta) * z2)

def combine_images(images, rows, cols, gap):
    width, height = images[0].size
    combined_width = cols * width
    combined_height = rows * height + (rows - 1) * gap
    combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            combined.paste(images[idx], (j * width, i * (height + gap)))
    return combined

def gen(image):
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise = model(image, t).sample
            image = scheduler.step(noise, t, image).prev_sample
        return image


def opt(gt_image, lr=0.003, num_iter=1000):
    pred_image = gt_image.detach().clone()
    pred_image.requires_grad=True
    optimizer = torch.optim.Adam([pred_image], lr=lr)
    for _ in range(num_iter):
        noise = model(pred_image, 0).sample
        image = scheduler.step(noise, 0, pred_image).prev_sample
        loss = F.mse_loss(image, gt_image)
        #print(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return pred_image

def fp_iter(image_20):
    with torch.no_grad():
        image = image_20
        for t in inv_scheduler.timesteps:
            if t==0:
                continue
            else:
                x = image
                for i in range(10):
                    noise = model(image, t).sample
                    image = inv_scheduler.step(noise, t, x).prev_sample
        return image

#init_noise = torch.load('init_latent.pt')
init_noise_1 = torch.randn(1,3,256,256).to('cuda')
gt_image = gen(init_noise_1)
image_20 = opt(gt_image)
print('optimization over')
pred_init_noise_1 = fp_iter(image_20)
print('fixed-point iteration over')
torch.save(pred_init_noise_1,'pred_noise_1.pt')

init_noise_2 = torch.randn(1,3,256,256).to('cuda')
gt_image = gen(init_noise_2)
image_20 = opt(gt_image)
print('optimization over')
pred_init_noise_2 = fp_iter(image_20)
print('fixed-point iteration over')
torch.save(pred_init_noise_2,'pred_noise_2.pt')

images = []
with torch.no_grad():
    for i in range(5):
        s_noise = slerp(init_noise_1, pred_init_noise_2, i/4)
        e_noise = slerp(init_noise_2, pred_init_noise_1, i/4)
        for j in range(5):
            image = slerp(s_noise, e_noise, j/4)
            for t in scheduler.timesteps:
                noise = model(image, t).sample
                image = scheduler.step(noise, t, image).prev_sample
            image = latent2image(image)
            images.append(image)
combine_images(images, 5, 5, 10).save('5times5.png')
