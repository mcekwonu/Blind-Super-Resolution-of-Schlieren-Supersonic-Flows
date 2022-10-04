import random
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from realesrgan_x8 import Generator, UNetDiscriminator, PerceptualLoss


# =======================================================================================
#                       Common configuration
# =======================================================================================
warnings.filterwarnings("ignore")
SEED = 112
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.benchmark = True
DEVICE = "cuda"


# =======================================================================================
#                       Training configuration
# =======================================================================================
# configure dataset
DATA_ROOT = "../../training_images_x8"
BATCH_SIZE = 32
LEARNING_RATE_PSNR = 2e-4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
NUM_IMAGES = BATCH_SIZE
NUM_WORKERS = 4
SAMPLE_INTERVAL = 100
WARMUP_ITERATIONS = 100_000
FILENAME = "RealESRGAN_x8"
RESIZE = False

# HR dimension
hr_shape = (1, 128, 128)

# configure WEIGHTS
generator = Generator(upsample_factor=8).to(DEVICE)
discriminator = UNetDiscriminator(input_shape=hr_shape).to(DEVICE)

# Optimizer
OPTIMIZER_PSNR = optim.Adam(generator.parameters(), lr=LEARNING_RATE_PSNR, betas=(0.9, 0.99))
OPTIMIZER_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
OPTIMIZER_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))


# Learning rate scheduler
milestones = [NUM_EPOCHS * 0.25, NUM_EPOCHS * 0.5, NUM_EPOCHS * 0.75]
LR_SCHEDULER_D = MultiStepLR(OPTIMIZER_D, list(map(int, milestones)), 0.5)
LR_SCHEDULER_G = MultiStepLR(OPTIMIZER_G, list(map(int, milestones)), 0.5)


# Loss functions
GAN_criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)
content_criterion = torch.nn.L1Loss().to(DEVICE)
perceptual_criterion = PerceptualLoss().to(DEVICE)
psnr_criterion = torch.nn.L1Loss().to(DEVICE)

# Training log
G_losses = list()
D_losses = list()
losses = OrderedDict()

# logs directories
IMAGES_DIR = "../training_logs/images"
CHECKPOINT_DIR = "../training_logs/checkpoints"
WEIGHTS_DIR = "../training_logs/weights"
HISTORY_DIR = "../training_logs/history"


# image writer
WRITER_HR = SummaryWriter(f"{IMAGES_DIR}/hr")
WRITER_SR = SummaryWriter(f"{IMAGES_DIR}/sr")

# Plot losses
LOSS_DICT = {"D_label": "D", "G_label": "G", "verbose": True,
             "title": "RealESRGAN_x8 Generator and Discriminator Loss",
             "output_dir": HISTORY_DIR,
             "date": datetime.now().strftime("%Y%m%d")
             }
