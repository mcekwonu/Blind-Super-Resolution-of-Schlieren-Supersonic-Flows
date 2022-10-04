import os
import torch
from collections import OrderedDict


def net_interpolate(weights_dir='training_logs/weights', alpha=0.8):
    net_ERSNet_path = os.path.join(weights_dir, "esrnet_x8.pth")
    net_ESRGAN_path = os.path.join(weights_dir, "esrgan_x8.pth")
    net_interp_path = os.path.join(weights_dir, "interp_{:d}.pth".format(int(alpha * 10)))

    net_ESRNet = torch.load(net_ERSNet_path)
    net_ESRGAN = torch.load(net_ESRGAN_path)
    net_interp = OrderedDict()

    print("Interpolating with alpha = ", alpha)

    for k, v_PSNR in net_ESRNet.items():
        v_ESRGAN = net_ESRGAN[k]
        net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

    torch.save(net_interp, net_interp_path)


if __name__ == "__main__":
    import argparse

    parser = ArgumentParser(description="Network Interpolation of PSNR and GAN-based")
    parser.add_argument('--weights_dir', '-w', type=str, required=True, help='saved models directory')
    parser.add_argument('--alpha', '-a', type=float, default=0.8, required=True, help='Interpolation parameter')
    opt = parser.parse_args()

    if opt:
        net_interpolate(opt.weights_dir, opt.alpha)
    else:
        net_interpolate()
