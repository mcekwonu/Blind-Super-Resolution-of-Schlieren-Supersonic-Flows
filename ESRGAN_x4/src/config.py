import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from esrgan_x4 import Generator, Discriminator, VGGLoss

# =======================================================================================
#                       Common configuration
# =======================================================================================
SEED = 142
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.benchmark = True
DEVICE = "cuda"

# =======================================================================================
#                       Training configuration
# =======================================================================================
# configure dataset
DATA_ROOT = "../../training_images_x4"
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
NUM_IMAGES = BATCH_SIZE
NUM_WORKERS = 4
SAMPLE_INTERVAL = 100
WARMUP_ITERATIONS = 100_000
FILENAME = "esrgan_x4"
RESIZE = False

# HR dimension
hr_shape = (1, 64, 64)

# configure model
generator = Generator(upsample_factor=4).to(DEVICE)
discriminator = Discriminator(input_shape=hr_shape, upsample_factor=4).to(DEVICE)

# Optimizer
OPTIMIZER_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
OPTIMIZER_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))

# Learning rate scheduler
milestones = [NUM_EPOCHS * 0.25, NUM_EPOCHS * 0.5, NUM_EPOCHS * 0.75]
LR_SCHEDULER_D = MultiStepLR(OPTIMIZER_D, list(map(int, milestones)), 0.5)
LR_SCHEDULER_G = MultiStepLR(OPTIMIZER_G, list(map(int, milestones)), 0.5)

# Loss functions
adversarial_criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)
content_criterion = torch.nn.L1Loss().to(DEVICE)
pixel_criterion = VGGLoss().to(DEVICE)
psnr_criterion = torch.nn.MSELoss().to(DEVICE)

# Training log
G_losses = list()
D_losses = list()
losses = OrderedDict()

# logs directories
IMAGES_DIR = "../training_logs/images"
CHECKPOINT_DIR = "../training_logs/checkpoints"
MODEL_DIR = "../training_logs/weights"
HISTORY_DIR = "../training_logs/history"

# image writer
WRITER_HR = SummaryWriter(f"{IMAGES_DIR}/hr")
WRITER_SR = SummaryWriter(f"{IMAGES_DIR}/sr")

# Plot losses
LOSS_DICT = {"D_label": "D", "G_label": "G",
             "verbose": True,
             "title": "ESRGAN_x4 Generator and Discriminator Loss",
             "output_dir": HISTORY_DIR,
             "date": datetime.now().strftime("%Y%m%d"),
             }
