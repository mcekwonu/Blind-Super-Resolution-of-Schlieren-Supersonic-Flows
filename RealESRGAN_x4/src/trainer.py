import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import SRDataset
import config
import utils


class Trainer(nn.Module):
    """Train RealESRGAN with pixel-loss, content-loss and adversarial loss.
    """

    def __init__(self):
        super().__init__()
        self.model_name = config.FILENAME
        self.data_root = config.DATA_ROOT
        utils.create_dir(config.CHECKPOINT_DIR)
        utils.create_dir(config.WEIGHTS_DIR)
        utils.create_dir(config.HISTORY_DIR)

    @property
    def _train(self):
        date = datetime.now().strftime("%Y%m%d")

        try:
            print("\nTraining RealESRGAN x4 ...\n")
            start_time = time.time()

            for epoch in range(config.NUM_EPOCHS):
                for step, (lr, hr) in enumerate(self.get_loader):
                    batches = len(self.get_loader)
                    iterations = epoch * batches + step

                    lr = lr.to(config.DEVICE)
                    hr = hr.to(config.DEVICE)

                    # Adversarial ground truths
                    real = torch.ones((lr.size(0), *config.discriminator.input_shape)).to(config.DEVICE)
                    fake = torch.ones((lr.size(0), *config.discriminator.input_shape)).to(config.DEVICE)

                    ##################################################################################
                    # Train Generator
                    ##################################################################################
                    config.OPTIMIZER_PSNR.zero_grad()

                    gen_hr = config.generator(lr)
                    psnr_loss = config.psnr_criterion(gen_hr, hr)

                    if iterations < config.WARMUP_ITERATIONS:
                        psnr_loss.backward()
                        config.OPTIMIZER_PSNR.step()
                        if iterations % config.SAMPLE_INTERVAL == 0:
                            print(
                                "[Epoch {}/{}] [Iterations: {}]\t[G psnr: {:.6f}]".format(
                                    epoch + 1, config.NUM_EPOCHS, iterations, psnr_loss.item())
                            )
                        self.save_model_weights(model=config.generator, filename=f"{config.WEIGHTS_DIR}/RealESRNet_x4.pth")
                        continue

                    config.OPTIMIZER_G.zero_grad()
                    pred_real = config.discriminator(hr).detach()
                    pred_fake = config.discriminator(gen_hr)

                    adv_loss = (config.GAN_criterion(pred_real - pred_fake.mean(0, keepdim=True), fake)
                                + config.GAN_criterion(pred_fake - pred_real.mean(0, keepdim=True), real)
                                ) / 2
                    content_loss = config.content_criterion(gen_hr, hr)
                    perceptual_loss = config.perceptual_criterion(gen_hr, hr)
                    loss_G = content_loss + perceptual_loss + 0.1 * adv_loss

                    loss_G.backward()
                    config.OPTIMIZER_G.step()
                    config.G_losses.append(loss_G.item())

                    ###################################################################################
                    # Train Discriminator
                    ###################################################################################
                    config.OPTIMIZER_D.zero_grad()

                    pred_real = config.discriminator(hr)
                    pred_fake = config.discriminator(gen_hr.detach())

                    loss_D = (config.GAN_criterion(pred_real - pred_fake.mean(0, keepdim=True), real)
                              + config.GAN_criterion(pred_fake - pred_real.mean(0, keepdim=True), fake)) / 2

                    loss_D.backward()
                    config.OPTIMIZER_D.step()
                    config.D_losses.append(loss_D.item())

                    ####################################################################################
                    # Log results: HR images, SR images, and D and G losses
                    ####################################################################################
                    if iterations % config.SAMPLE_INTERVAL == 0:
                        print(
                            "[Epoch {}/{}] [Iterations: {}] [D Loss: {:.5f}]  [G Loss: {:.5f},"
                            " content: {:.5f}, adv: {:.5f}, perceptual: {:.5f}]".format(
                                epoch + 1, config.NUM_EPOCHS, iterations, loss_D.item(), loss_G.item(),
                                content_loss.item(), adv_loss.item(), perceptual_loss.item()
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
                config.LR_SCHEDULER_G.step()
                config.LR_SCHEDULER_D.step()
            end_time = time.time() - start_time

            # save losses
            config.losses["G_losses"] = config.G_losses
            config.losses["D_losses"] = config.D_losses
            np.savez_compressed(f"{config.HISTORY_DIR}/losses_{date}", **config.losses)

            self.save_model_weights(
                model=config.generator,
                filename=f"{config.WEIGHTS_DIR}/{self.model_name}_{date}.pth"
            )
            utils.log_timer(f"{config.HISTORY_DIR}/training_time_{date}.txt", end_time)

            # Plot losses
            self.plot_loss(path=f"{config.HISTORY_DIR}/losses_{date}.npz", **config.LOSS_DICT)

        except KeyboardInterrupt:
            self.save_model_weights(
                model=config.generator,
                filename=f"{config.WEIGHTS_DIR}/{self.model_name}_{date}.pth"
            )

    def _resume(self, epoch, ckpt):
        """ Resume training from saved checkpoint. The last epoch and model weight and optimizer states
        are loaded.

        Parameters:
             epoch (int): Number of epoch to complete training
             ckpt (str):  Saved checkpoint file

        Returns:
            None
        """
        try:
            # configure model and optimizer
            generator, discriminator, optimizer_G, optimizer_D, resume_epoch, LR = self.load_checkpoint(
                checkpoint_file=f"{config.CHECKPOINT_DIR}/checkpoint_{ckpt}.pth.tar",
                model_A=config.generator,
                model_B=config.discriminator,
                optimizer_A=config.OPTIMIZER_G,
                optimizer_B=config.OPTIMIZER_D,
                device=config.DEVICE,
            )
            OPTIMIZER_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.99))
            OPTIMIZER_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.9, 0.99))

            date = datetime.now().strftime("%Y%m%d")
            print(f"\nResuming RealESRGAN x4 training from last checkpoint ({resume_epoch} epoch)...\n")
            start_timer = time.time()

            for n_epoch in range(resume_epoch, resume_epoch + epoch):
                for step, (lr, hr) in enumerate(self.get_loader):
                    batches = len(self.get_loader)
                    iterations = epoch * batches + step

                    lr = lr.to(config.DEVICE)
                    hr = hr.to(config.DEVICE)

                    # Adversarial ground truths
                    real = torch.ones((lr.size(0), *config.discriminator.input_shape)).to(config.DEVICE)
                    fake = torch.ones((lr.size(0), *config.discriminator.input_shape)).to(config.DEVICE)

                    ##################################################################################
                    # Train Generator
                    ##################################################################################
                    OPTIMIZER_G.zero_grad()

                    gen_hr = generator(lr)
                    pred_real = discriminator(hr).detach()
                    pred_fake = discriminator(gen_hr)

                    adv_loss = (config.GAN_criterion(pred_real - pred_fake.mean(0, keepdim=True), fake)
                                + config.GAN_criterion(pred_fake - pred_real.mean(0, keepdim=True), real)
                                ) / 2
                    content_loss = config.content_criterion(gen_hr, hr)
                    perceptual_loss = config.perceptual_criterion(gen_hr, hr)
                    loss_G = content_loss + perceptual_loss + 0.1 * adv_loss

                    loss_G.backward()
                    OPTIMIZER_G.step()
                    config.G_losses.append(loss_G.item())

                    ###################################################################################
                    # Train Discriminator
                    ###################################################################################
                    OPTIMIZER_D.zero_grad()

                    pred_real = discriminator(hr)
                    pred_fake = discriminator(gen_hr.detach())

                    loss_D = (config.GAN_criterion(pred_real - pred_fake.mean(0, keepdim=True), real)
                              + config.GAN_criterion(pred_fake - pred_real.mean(0, keepdim=True), fake)) / 2

                    loss_D.backward()
                    OPTIMIZER_D.step()
                    config.D_losses.append(loss_D.item())

                    ####################################################################################
                    # Log results: HR images, SR images, and D and G losses
                    ####################################################################################
                    if iterations % config.SAMPLE_INTERVAL == 0:
                        print(
                            "[Epoch {}/{}] [Iterations: {}]  [D Loss: {:.5f}]  [G Loss: {:.5f},"
                            " content: {:.5f}, adv: {:.5f}, perceptual: {:.5f}]".format(
                                n_epoch, resume_epoch + epoch, iterations, loss_D.item(), loss_G.item(),
                                content_loss.item(), adv_loss.item(), perceptual_loss.item()
                            )
                        )

                        with torch.no_grad():
                            sr = generator(lr)
                        img_grid_hr = make_grid(hr[:config.NUM_IMAGES], normalize=True)
                        img_grid_sr = make_grid(sr[:config.NUM_IMAGES], normalize=True)
                        config.WRITER_HR.add_image("High Resolution", img_grid_hr, global_step=iterations)
                        config.WRITER_SR.add_image("Super Resolution", img_grid_sr, global_step=iterations)

                        self.save_checkpoints(
                            epoch=n_epoch + 1,
                            model_A=config.generator,
                            model_B=config.discriminator,
                            optimizer_A=config.OPTIMIZER_G,
                            optimizer_B=config.OPTIMIZER_D,
                            filename=f"{config.CHECKPOINT_DIR}/checkpoint_{date}.pth.tar"
                        )
                config.LR_SCHEDULER_G.step()
                config.LR_SCHEDULER_D.step()
            end_timer = time.time() - start_timer

            # save losses
            config.losses["G_losses"] = config.G_losses
            config.losses["D_losses"] = config.D_losses
            np.savez_compressed(f"{config.HISTORY_DIR}/losses_{date}", **config.losses)

            self.save_model_weights(
                model=config.generator,
                filename=f"{config.WEIGHTS_DIR}/{self.model_name}_{date}.pth"
            )
            utils.log_timer(f"{config.HISTORY_DIR}/training_time_{date}.txt", end_timer)

            # Plot losses
            self.plot_loss(path=f"{config.HISTORY_DIR}/losses_{date}.npz", **config.LOSS_DICT)

        except KeyboardInterrupt:
            self.save_model_weights(
                model=config.generator,
                filename=f"{config.WEIGHTS_DIR}/{self.model_name}_{date}.pth"
            )

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
            "optimizer_B_state_dict": optimizer_B.state_dict(),
        }
        torch.save(states, filename)

    def load_checkpoint(self, checkpoint_file, model_A, model_B, optimizer_A, optimizer_B, device="cuda"):
        print(" Loading checkpoint ...")
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
    def train_pipeline():
        Trainer()._train

    @staticmethod
    def resume_pipeline(epoch, ckpt):
        Trainer()._resume(epoch=epoch, ckpt=ckpt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python trainer.py", description="Super Resolution RealESRGAN x4",
                                     usage="%(prog)s [options]")
    parser.add_argument("--train", action='store_true', help="Train RealESRGAN x4.[option] [filename]")
    parser.add_argument("--resume", action='store_true',
                        help="Resume/finetune RealESRGAN x4. [options] [filename, epoch, ckpt]")
    parser.add_argument("--epoch", "-e", type=int, help="Number of epoch")
    parser.add_argument("--ckpt", "-c", type=str, help="Saved checkpoint")

    opt = parser.parse_args()

    EPOCH = opt.epoch
    CHECKPOINTS = opt.ckpt

    if opt.train:
        Trainer.train_pipeline()
    elif opt.resume:
        Trainer.resume_pipeline(epoch=EPOCH, ckpt=CHECKPOINTS)
    else:
        parser.parse_args(["-h"])
