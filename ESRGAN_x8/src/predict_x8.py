import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
from tqdm.auto import tqdm

from esrgan_x8 import Generator


def predict(input_dir, model_path, save_dir, device="cpu", upscale_factor=8, num_images=-1, verbose=True):
    """Inference using ESRGAN x4 trained model

        Parameters:
            input_dir: (str) Test image directory
            save_dir: (str) Output save directory for super resolved images
            model_path: (str) Saved location of model weights
            device: (str) `CPU` or `GPU`, default=cpu
            upscale_factor: Upscale factor, default=8
            num_images: (int) Number of images to process
            verbose: (bool) to save processing time of images
    """
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ])

    model = Generator(upsample_factor=upscale_factor)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)
    time_completed = []

    begin = time.time()
    print()

    for root, dirs, files in os.walk(input_dir):
        for idx, file in tqdm(
                enumerate(sorted(files)), desc="Running super resolution using ESRGAN x8",
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
                time_completed.append(f"{fname}.png completed in {elapsed // 60:.0f}m, {elapsed % 60:.0f}s.\n")

                if verbose:
                    with open(f"{save_dir}/time_logs.txt", "w+") as f:
                        f.write(f"Time log for super resolution x8 of images {os.path.basename(input_dir)}\n")
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

    parser = argparse.ArgumentParser(prog="python predict_x8.py", description="Super Resolution with ESRGAN x8",
                                     usage="%(prog)s [options]")
    parser.add_argument("--input_dir", "-i", type=str, help="Low resolution images directory")
    parser.add_argument("--output_dir", "-o", type=str, help="Directory to save super resolved images")
    parser.add_argument("--model_dir", "-m", type=str, help="Saved model weights directory")
    parser.add_argument("--scale", "-s", type=int, default=8, help="Spatial upscaling factor")
    parser.add_argument("--device", "-d", default="cpu", type=str, choices=["cpu", "cuda"], help="using CPU or GPU")
    parser.add_argument("--num_images", "-n", default=-1, type=int,
                        help="Number of images to save. If set to -1, all images are saved. Default = -1")
    parser.add_argument("--verbose", "-v", type=bool, default=True, help="to save processing time of each image")

    opt = parser.parse_args()

    DEVICE = opt.device
    INPUT_DIR = opt.input_dir
    MODEL_DIR = opt.model_dir
    SAVE_DIR = opt.output_dir
    UPSCALE_FACTOR = opt.scale
    NUM_IMAGES = opt.num_images
    VERBOSE = opt.verbose

    predict(input_dir=INPUT_DIR, model_path=MODEL_DIR, save_dir=SAVE_DIR, device=DEVICE, upscale_factor=UPSCALE_FACTOR,
            num_images=NUM_IMAGES, verbose=VERBOSE)
