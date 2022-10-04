import time
import os
import torch
import warnings
import numpy as np
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import SRDataset
import config
import utils

warnings.filterwarnings("ignore")


class Trainer:
    """Train ESRGAN with pixel-loss, content-loss and adversarial loss. And
    _train only SRResNet (Generator) with pixel-loss (L2 loss)

    methods:
    _train
    _resume
    """

    def __init__(self):
        self.model_name = config.FILENAME
        self.data_root = config.DATA_ROOT
        utils.create_dir(config.CHECKPOINT_DIR)
        utils.create_dir(config.MODEL_DIR)
        utils.create_dir(config.HISTORY_DIR)

    def _train(self):
        date = datetime.now().strftime("%Y%m%d")

        try:
            print("\nStarting ESRGAN x8 Training...\n")
            start_time = time.time()

            for epoch in range(config.NUM_EPOCH):
                for step, (lr, hr) in enumerate(self.get_loader):
                    batches = len(self.get_loader)
                    iterations = epoch * batches + step

                    lr = lr.to(config.DEVICE)
                    hr = hr.to(config.DEVICE)

                    # Adversarial ground truths
                    real = torch.ones((lr.size(0), *config.discriminator.output_shape)).to(config.DEVICE)
                    fake = torch.ones((lr.size(0), *config.discriminator.output_shape)).to(config.DEVICE)

                    ##################################################################################
                    # Train Generator: maximize log(D(G(z)))
                    ##################################################################################
                    config.OPTIMIZER_PSNR.zero_grad()

                    gen_hr = config.generator(lr)
                    psnr_loss = config.psnr_criterion(gen_hr, hr)

                    if iterations < config.WARMUP_ITERATIONS:
                        psnr_loss.backward()
                        config.OPTIMIZER_PSNR.step()
                        if iterations % config.SAMPLE_INTERVAL == 0:
                            print(
                                "[Epoch {}/{}] [Iterations: {}]\t[G psnr: {:.5}]".format(
                                    epoch + 1, config.NUM_EPOCH, iterations + 1, psnr_loss.item())
                            )
                        self.save_model_weights(model=config.generator, filename=f"{config.WEIGHTS_DIR}/ESRNet_x8.pth")
                        continue

                    config.OPTIMIZER_G.zero_grad()
                    pixel_loss = config.pixel_criterion(gen_hr, hr)
                    pred_real = config.discriminator(hr).detach()
                    pred_fake = config.discriminator(gen_hr)

                    adv_loss = config.adversarial_criterion(pred_fake - pred_real.mean(0, keepdim=True), real)
                    content_loss = config.content_criterion(gen_hr, hr)
                    loss_G = content_loss + 5e-3 * adv_loss + 1e-2 * pixel_loss

                    loss_G.backward()
                    config.OPTIMIZER_G.step()
                    config.G_losses.append(loss_G.item())

                    ###################################################################################
                    # Train Discriminator: maximize log(D(x)) + (1 - log(D(G(z))))
                    ###################################################################################
                    config.OPTIMIZER_D.zero_grad()

                    pred_real = config.discriminator(hr)
                    pred_fake = config.discriminator(gen_hr.detach())

                    loss_D = (config.adversarial_criterion(pred_real - pred_fake.mean(0, keepdim=True), real)
                              + config.adversarial_criterion(pred_fake - pred_real.mean(0, keepdim=True), fake)) / 2

                    loss_D.backward()
                    config.OPTIMIZER_D.step()
                    config.D_losses.append(loss_D.item())

                    ####################################################################################
                    # Log results: HR images, SR images, and D and G losses
                    ####################################################################################
                    if iterations % config.SAMPLE_INTERVAL == 0:
                        print(
                            "[Epoch {}/{}] [Iterations: {}] [D Loss: {:.5f}]  [G Loss: {:.5f},"
                            " content: {:.5}, adv: {:.5}, pixel: {:.5}]".format(
                                epoch + 1, config.NUM_EPOCH, iterations + 1, loss_D.item(), loss_G.item(),
                                content_loss.item(), adv_loss.item(), pixel_loss.item()
                            )
                        )

                        with torch.no_grad():
                            sr = config.generator(lr)
                        img_grid_hr = make_grid(hr[:config.NUM_IMAGES], normalize=True)
                        img_grid_sr = make_grid(sr[:config.NUM_IMAGES], normalize=True)
                        config.WRITER_HR.add_image("High Resolution", img_grid_hr, global_step=iterations)
                        config.WRITER_SR.add_image("Super Resolution", img_grid_sr, global_step=iterations)

                        self.save_checkpoints(
                            epoch=epoch + 1,
                            model_A=config.generator,
                            model_B=config.discriminator,
                            optimizer_A=config.OPTIMIZER_G,
                            optimizer_B=config.OPTIMIZER_D,
                            filename=f"{config.CHECKPOINT_DIR}/checkpoint_{date}.pth.tar"
                        )
                config.LR_SCHEDULER_D.step()
                config.LR_SCHEDULER_G.step()
            end_time = time.time() - start_time

            # save losses
            config.losses["G_losses"] = config.G_losses
            config.losses["D_losses"] = config.D_losses
            np.savez_compressed(f"{config.HISTORY_DIR}/losses_{date}", **config.losses)

            self.save_model_weights(model=config.generator, filename=f"{config.MODEL_DIR}/{self.model_name}_{date}.pth")
            utils.log_timer(f"{config.HISTORY_DIR}/training_time_{date}.txt", end_time)

            # Plot losses
            self.plot_loss(path=f"{config.HISTORY_DIR}/losses_{date}.npz", **config.LOSS_DICT)

        except KeyboardInterrupt:
            self.save_model_weights(model=config.generator, filename=f"{config.MODEL_DIR}/{self.model_name}_{date}.pth")

    def _resume(self, epoch, ckpt):
        """
        Resume training from last saved checkpoint.

        Parameters:
            epoch: (int) Number of epoch to train
		    ckpt: (str) Saved checkpoint
		"""
        try:
            # configure model and optimizer
            generator, discriminator, optimizer_G, optimizer_D, resume_epoch, LR = self.load_checkpoint(
                checkpoint_file=f"{config.CHECKPOINT_DIR}/checkpoint_{ckpt}.pth.tar",
                model_A=config.generator,
                model_B=config.discriminator,
                optimizer_A=config.OPTIMIZER_G,
                optimizer_B=config.OPTIMIZER_D,
                device=config.DEVICE
            )
            OPTIMIZER_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.99))
            OPTIMIZER_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.9, 0.99))

            date = datetime.now().strftime("%Y%m%d")
            print(f"\nResuming ESRGAN x8 training from last saved checkpoint with {resume_epoch} epochs...\n")
            start_timer = time.time()

            for num_epoch in range(resume_epoch, resume_epoch + epoch):
                for step, (lr, hr) in enumerate(self.get_loader):
                    batches = len(self.get_loader)
                    iterations = num_epoch * batches + step

                    lr = lr.to(config.DEVICE)
                    hr = hr.to(config.DEVICE)

                    # Adversarial ground truths
                    real = torch.ones((lr.size(0), *discriminator.output_shape)).to(config.DEVICE)
                    fake = torch.ones((lr.size(0), *discriminator.output_shape)).to(config.DEVICE)

                    ##################################################################################
                    # Train Generator: maximize log(D(G(z)))
                    ##################################################################################
                    OPTIMIZER_G.zero_grad()

                    gen_hr = generator(lr)
                    pixel_loss = config.pixel_criterion(gen_hr, hr)

                    pred_real = discriminator(hr).detach()
                    pred_fake = discriminator(gen_hr)

                    adv_loss = config.adversarial_criterion(pred_fake - pred_real.mean(0, keepdim=True), real)
                    content_loss = config.content_criterion(gen_hr, hr)
                    loss_G = content_loss + 5e-3 * adv_loss + 1e-2 * pixel_loss

                    loss_G.backward()
                    OPTIMIZER_G.step()
                    config.G_losses.append(loss_G.item())

                    ###################################################################################
                    # Train Discriminator: maximize log(D(x)) + (1 - log(D(G(z))))
                    ###################################################################################
                    OPTIMIZER_D.zero_grad()

                    pred_real = discriminator(hr)
                    pred_fake = discriminator(gen_hr.detach())

                    loss_D = (config.adversarial_criterion(pred_real - pred_fake.mean(0, keepdim=True), real)
                              + config.adversarial_criterion(pred_fake - pred_real.mean(0, keepdim=True), fake)) / 2

                    loss_D.backward()
                    OPTIMIZER_D.step()
                    config.D_losses.append(loss_D.item())

                    ####################################################################################
                    # Log results: HR images, SR images, and D and G losses
                    ####################################################################################
                    if iterations % config.SAMPLE_INTERVAL == 0:
                        print(
                            "[Epoch {}/{}] [Iterations: {}] [D Loss: {:.5f}]  [G Loss: {:.5f},"
                            " content: {:.5}, adv: {:.5}, pixel: {:.5}]".format(
                                num_epoch, resume_epoch + epoch, iterations, loss_D.item(), loss_G.item(),
                                content_loss.item(), adv_loss.item(), pixel_loss.item()
                            )
                        )

                        with torch.no_grad():
                            sr = generator(lr)
                        img_grid_hr = make_grid(hr[:config.NUM_IMAGES], normalize=True)
                        img_grid_sr = make_grid(sr[:config.NUM_IMAGES], normalize=True)
                        config.WRITER_HR.add_image("High Resolution", img_grid_hr, global_step=iterations)
                        config.WRITER_SR.add_image("Super Resolution", img_grid_sr, global_step=iterations)

                        self.save_checkpoints(
                            epoch=num_epoch + 1,
                            model_A=generator,
                            model_B=discriminator,
                            optimizer_A=OPTIMIZER_G,
                            optimizer_B=OPTIMIZER_D,
                            filename=f"{config.CHECKPOINT_DIR}/checkpoint_{date}.pth.tar"
                        )

                config.LR_SCHEDULER_D.step()
                config.LR_SCHEDULER_G.step()
            end_timer = time.time() - start_timer

            # save losses
            config.losses["G_losses"] = config.G_losses
            config.losses["D_losses"] = config.D_losses
            np.savez_compressed(f"{config.HISTORY_DIR}/losses_{date}", **config.losses)

            self.save_model_weights(model=config.generator, filename=f"{config.MODEL_DIR}/{self.model_name}_{date}.pth")
            utils.log_timer(f"{config.HISTORY_DIR}/training_time_{date}.txt", end_timer, mode="a")

            # Plot losses
            self.plot_loss(path=f"{config.HISTORY_DIR}/losses_{date}.npz", **config.LOSS_DICT)

        except KeyboardInterrupt:
            self.save_model_weights(model=config.generator, filename=f"{config.MODEL_DIR}/{self.model_name}_{date}.pth")

    @property
    def get_loader(self):
        dataloader = DataLoader(
            SRDataset(root_dir=self.data_root, resize=config.RESIZE), batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True
        )
        return dataloader

    def save_model_weights(self, model, filename):
        torch.save(model.state_dict(), filename)

    def save_checkpoints(self, epoch, model_A, model_B, optimizer_A, optimizer_B, filename):
        states = {
            "epoch": epoch,
            "model_A_state_dict": model_A.state_dict(),
            "optimizer_A_state_dict": optimizer_A.state_dict(),
            "model_B_state_dict": model_B.state_dict(),
            "optimizer_B_state_dict": optimizer_B.state_dict()
        }
        torch.save(states, filename)

    def load_checkpoint(self, checkpoint_file, model_A, model_B, optimizer_A, optimizer_B, device="cuda"):
        print("Loading checkpoint ...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model_A.load_state_dict(checkpoint["model_A_state_dict"])
        optimizer_A.load_state_dict(checkpoint["optimizer_A_state_dict"])
        model_B.load_state_dict(checkpoint["model_B_state_dict"])
        optimizer_B.load_state_dict(checkpoint["optimizer_B_state_dict"])
        resume_epoch = checkpoint["epoch"]
        for param_group in optimizer_A.param_groups:
            lr = param_group["lr"]

        return model_A, model_B, optimizer_A, optimizer_B, resume_epoch, lr

    def plot_loss(self, path, **kwargs):
        losses = np.load(path)
        g_losses, d_losses = losses['G_losses'], losses['D_losses']

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.plot(g_losses, label=kwargs["G_label"], color='red')
        ax.plot(d_losses, label=kwargs["D_label"], color='green')
        ax.set(xlabel="Iterations", ylabel="Loss", title=kwargs["title"])
        ax.legend(frameon=False, loc="best")
        if kwargs["verbose"]:
            fig.savefig(kwargs["output_dir"] + "/loss_" + kwargs["date"] + ".png", bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def train_esrgan():
        Trainer()._train()

    @staticmethod
    def resume_esrgan(epoch, ckpt):
        Trainer()._resume(epoch=epoch, ckpt=ckpt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python trainer.py", description="Super Resolution ESRGAN x8",
                                     usage="%(prog)s [options]")
    parser.add_argument("--train", action='store_true', help="Train ESRGAN x8.[option] [filename]")
    parser.add_argument("--resume", action='store_true',
                        help="Resume/finetune ESRGAN x8. [options] [epoch, ckpt (enter the date)]")
    parser.add_argument("--epoch", "-e", type=int, help="Number of epoch")
    parser.add_argument("--ckpt", "-c", type=str, help="Saved checkpoint date")

    opt = parser.parse_args()

    EPOCH = opt.epoch
    CHECKPOINTS = opt.ckpt

    if opt.train:
        Trainer.train_esrgan()
    elif opt.resume:
        Trainer.resume_esrgan(epoch=EPOCH, ckpt=CHECKPOINTS)
    else:
        parser.parse_args(["-h"])
