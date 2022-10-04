import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from IPython.display import HTML
from torchvision.utils import make_grid


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


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


def make_animation(images_list):
    fig = plt.figure(figsize=(16, 16))
    plt.axis("off")
    imgs = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in images_list]
    ani = animation.ArtistAnimation(fig, imgs, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())


def show_real_and_fake_images(dataloader, imgs_list, num_images, save_dir):
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    real_img = np.transpose(
        make_grid(real_batch[0].to(device)[:num_images], padding=1, normalize=True).cpu(),
        (1, 2, 0))
    plt.imshow(real_img)

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(imgs_list[-1][:64], (1, 2, 0)))
    plt.savefig(f"{save_dir}/real_fake_images.png", bbox_inches="tight")
    plt.show()
