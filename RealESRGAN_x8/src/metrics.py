import os
import cv2
import glob
import math
import csv
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.ndimage import convolve
from scipy.special import gamma
from tqdm.auto import tqdm


def ssim(sr, hr):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.dot(kwernel, kernel.transpose())

    mu_x = cv2.filter2D(sr, -1, window)[5:-5, 5:-5]
    mu_y = cv2.filter2D(hr, -1, window)[5:-5, 5:-5]

    sigma_x_sq = cv2.filter2D(sr ** 2, -1, window)[5:-5, 5:-5] - mu_x ** 2
    sigma_y_sq = cv2.filter2D(hr ** 2, -1, window)[5:-5, 5:-5] - mu_y ** 2
    sigma = cv2.filter2D(sr * hr, -1, window)[5:-5, 5:-5] - (mu_x * mu_y)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    return ssim_map


def compute_ssim(sr_path: str, hr_path: str) -> float:
    """Compute SSIM between SR and HR"""
    sr = cv2.imread(sr_path)
    hr = cv2.imread(hr_path)

    if not sr.shape == hr.shape:
        raise ValueError("Input images must have the same dimension!")

    if sr.ndim == 2:
        return ssim(sr, hr)

    elif sr.ndim == 3:
        if sr.shape[-1] == 3:
            ssims = []
            for _ in range(sr.shape[-1]):
                ssims.append(ssim(sr, hr))
                return np.array(ssims).mean()

        else:
            raise ValueError(
                f"Wrong input image dimensions. Dimension should be 2 or 3 and not {sr.ndim}"
            )


def compute_psnr(sr_path: str, hr_path: str) -> float:
    """Compute Peak Signal to Noise Ratio (PSNR)"""

    sr = cv2.imread(sr_path)
    hr = cv2.imread(hr_path)

    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return float("inf")

    return 20 * math.log10(255. / math.sqrt(mse))


def compute_metrics(sr_path: str, hr_path: str) -> (float, float):
    psnr = compute_psnr(sr_path, hr_path)
    ssim = compute_ssim(sr_path, hr_path)

    return psnr, ssim


def FID(x, g):
    """Computes the Frechet Inception Distance"""

    inceptionv3 = inception_v3(pretrained=True)
    features = nn.Sequential(*list(inceptionv3.children())[:7])

    for params in features.parameters():
        params.requires_grad = False

    hr_features = features(x)
    sr_features = features(g)
    hr_mean = torch.mean(hr_features)
    sr_mean = torch.mean(sr_features)
    hr_cov = torch.cov(hr_features.view(-1, hr_features.size(3)))
    sr_cov = torch.cov(sr_features.view(-1, sr_features.size(3)))

    fid = torch.abs(hr_mean - sr_mean) ** 2 + torch.trace(hr_cov + sr_cov - 2 * torch.sqrt(hr_cov * sr_cov))
    return fid


def compute_FID(hr_path, sr_path):
    sr = cv2.imread(sr_path)
    hr = cv2.imread(hr_path)
    hr = torch.tensor(hr.transpose(2, 0, 1)[np.newaxis, :], dtype=torch.float32)
    sr = torch.tensor(sr.transpose(2, 0, 1)[np.newaxis, :], dtype=torch.float32)

    return FID(hr, sr)


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Parameters:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD distribution
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))
    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gamma_hat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block ** 2)
    rhat_norm = (rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2)
    array_position = np.argmin((r_gam - rhat_norm) ** 2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))

    return alpha, beta_l, beta_r


def compute_feature(block):
    """Compute VGGFeaturesExtractor.

    Parameters:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])

    return feat


def imresize(img, scale):
    """resize image using cv2.

    Parameters:
        img (ndarray): Input image
        scale (float): scale to reduce image
    """
    h, w = img.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)

    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return out


def reorder_image(img, input_order="HWC"):
    """Reorder images to HWC (Height, Width, Channel) order.

    Ref: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/metric_util.py

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Parameters:
        img (ndarray): Input image
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if img.ndim == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)

    return img


def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """Computes the NIQE (Natural Image Quality Evaluator) metric.

    Parameters:
        img (ndarray): Input image whose quality needs to be computed. The image must be in grayscale
                        with shape (h, w). Range [0, 255] with float type
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian model calculated on the
                        pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate Gaussian model calculated on
                        the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the image.
        block_size_h (int): Height of the blocks into which image is divided. Default is 96 (recommended value).
        block_size_w (int): Width of the blocks into which image is divided. Default is 96 (recommended value).
    """
    assert img.ndim == 2, f"Input image must be a grayscale image with shape (h, w)."

    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    dist_param = []
    for scale in [1, 2]:
        mu = convolve(img, gaussian_window, mode="nearest")
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode="nearest") - np.square(mu)))
        img_norm = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_norm[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                        idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))

        dist_param.append(np.array(feat))

        if scale == 1:
            img = imresize(img, scale=0.5)

    dist_param = np.concatenate(dist_param, axis=1)
    mu_dist_param = np.nanmean(dist_param, axis=0)
    dist_param_no_nan = dist_param[~np.isnan(dist_param).any(axis=1)]
    cov_dist_param = np.cov(dist_param_no_nan, rowvar=False)

    # compute niqe quality
    inv_cov_param = np.linalg.pinv((cov_pris_param + cov_dist_param) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_dist_param), inv_cov_param), np.transpose((mu_pris_param - mu_dist_param))
    )
    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))

    return quality


def compute_niqe(img, crop_border=0, **kwargs):
    """Compute NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.

    Parameters:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
        crop_border (int): Cropped pixels in each edge of an image.
            These pixels are not involved in the metric calculation

    Returns:
        float: NIQE result.
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    niqe_pris_params = np.load(os.path.join(ROOT_DIR, "../../metrics/niqe_pris_params.npz"))
    mu_pris_param = niqe_pris_params["mu_pris_param"]
    cov_pris_param = niqe_pris_params["cov_pris_param"]
    gaussian_window = niqe_pris_params["gaussian_window"]

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.round()
    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result


def main(source_dir):
    """Computes NIQE scores for all image in specified folder"""
    filename = os.path.basename(source_dir)
    img_paths = (os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith("png"))
    img_paths = sorted(img_paths)
    target_dir = os.path.abspath(source_dir)
    scores = []

    for img_path in tqdm(img_paths, desc="Computing NIQE score"):
        img = cv2.imread(img_path)
        niqe_score = compute_niqe(img, crop_border=0)
        scores.append(niqe_score)

    scores = np.asarray(scores).reshape(-1, 1)
    with open(f"{target_dir}/{filename}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(scores)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute NIQE score")
    parser.add_argument("--source_dir", "-i", type=str, required=True, help="Super resolved images")
    opt = parser.parse_args()

    SOURCE_DIR = opt.source_dir
    main(source_dir=SOURCE_DIR)


