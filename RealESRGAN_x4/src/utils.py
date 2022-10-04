import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from torchvision.utils import make_grid


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log_timer(path, time_elapsed, mode="w"):
    d, h, m, s = convert_to_time(time_elapsed)
    msg = f"Training completed in {h:g}hrs {m:g} mins\n"

    if mode == "w":
        with open(path, mode) as f:
            f.write(msg)
    elif mode == "a":
        with open(path, mode) as f:
            f.write(msg)
    else:
        raise NotImplemented(f"{mode} is Not Implemented!")


def convert_to_time(time):
    """Print the training time in days: hours: minutes: seconds format."""
    days = time // (24 * 60 * 60)
    time %= (24 * 60 * 60)
    hrs = time // (60 * 60)
    time %= (60 * 60)
    mins = time // 60
    time %= 60
    secs = time

    return days, hrs, mins, secs


def show_real_and_fake_images(dataloader, imgs_list):
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    real_img = np.transpose(
        make_grid(real_batch[0].to(device)[:config.NUM_IMAGES], padding=1, normalize=True).cpu(),
        (1, 2, 0))
    plt.imshow(real_img)

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(imgs_list[-1][:64], (1, 2, 0)))
    plt.savefig(f"{config.IMAGES_DIR}/real_fake_images.png", bbox_inches="tight")
    plt.show()
