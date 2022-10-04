import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
from tqdm.auto import tqdm

from realesrgan_x4 import Generator
from metrics import compute_niqe


def predict(input_dir, save_dir, model_path, device='cpu', upscale_factor=4, num_images=-1, verbose=True):
    """Inference using RealESRGAN x4 trained model

    Parameters:
        input_dir: (str) Test image directory
        save_dir: (str) Output save directory for super resolved images
        model_path: (str) Saved location of model weights
        device: (str) `CPU` or `GPU`, default=cpu
        upscale_factor: Upscale factor, default=2
        num_images: (int) Number of images to process
        verbose: (bool) to save processing time of images
    """
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])])

    model = Generator(upsample_factor=upscale_factor)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)
    time_completed = []

    begin = time.time()
    print()
    for root, dirs, files in os.walk(input_dir):
        for idx, file in tqdm(
                enumerate(sorted(files)), desc="Running super resolution using RealESRGAN x4",
                total=len(files)
        ):
            fname = os.path.splitext(file)[0]
            path = os.path.abspath(os.path.join(root, file))
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = test_transforms(img)
            img_LR = img.unsqueeze(0).to(device)

            with torch.no_grad():
                start = time.time()
                output = model(img_LR).detach().squeeze().float().cpu()
                output = output * 0.5 + 0.5
                save_image(output, f'{save_dir}/{fname}.png')
                elapsed = time.time() - start
                time_completed.append(f"{fname}.png completed in {elapsed % 60:.0f}s.\n")

                if verbose:
                    with open(f"{save_dir}/time_logs.txt", "w+") as f:
                        f.write(f"Time log for super resolution  of images for {os.path.basename(input_dir)}\n")
                        f.write(f"{'_' * 60}\n\n")
                        f.writelines(time_completed)

            if idx == num_images:
                break
            elif num_images == -1:
                continue
    ended = time.time() - begin
    print(f'Completed in {ended // 60:.0f}m, {ended % 60:.0f}s.\nSaved SR image in {os.path.abspath(save_dir)}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Super Resolution with RealESRGAN x4")
    parser.add_argument("--input_path", "-i", type=str, required=True, help="Low resolution images directory")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save super resolved images")
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="Saved model weight directory")
    parser.add_argument("--scale", "-s", type=int, default=4, help="Spatial upscaling factor")
    parser.add_argument("--device", "-d", default="cpu", type=str, choices=["cpu", "cuda"], help="GPU or CPU")
    parser.add_argument("--num", "-n", default=-1, type=int,
                        help="Number of images to process. If set to -1, all images are saved. Default = -1")
    parser.add_argument("--verbose", "-v", type=bool, default=True, help="to save processing time of each image")

    opt = parser.parse_args()

    DEVICE = opt.device
    INPUT_DIR = opt.input_dir
    MODEL_DIR = opt.model_dir
    UPSCALE_FACTOR = opt.scale
    SAVE_DIR = opt.save_dir
    NUM_IMAGES = opt.num
    VERBOSE = opt.verbose

    predict(input_dir=INPUT_DIR, save_dir=SAVE_DIR, model_path=MODEL_DIR, device=DEVICE, upscale_factor=UPSCALE_FACTOR,
            num_images=NUM_IMAGES, verbose=VERBOSE)
