import os
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union, Optional, List
from urllib.parse import urlparse

import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from diffusers import AutoencoderKL
from matplotlib.axes import Axes
from PIL import Image
from torchvision import transforms


def download_image(url: str, save_path: Optional[Union[str, Path]] = None) -> Image.Image:
    """
    Download an image from a URL and optionally save it.

    Args:
        url: The URL of the image to download
        save_path: Optional path to save the downloaded image

    Returns:
        The downloaded image as a PIL Image object

    Raises:
        Exception: If the download fails
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        if save_path:
            # Create parent directories if they don't exist
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)
        return img
    else:
        raise Exception(f"Failed to download image: {response.status_code}")


def resize_and_crop_square(image: Image.Image, target_size: int) -> Image.Image:
    """
    Resize an image and crop it to a square aspect ratio (1:1)
    with the target_size as both width and height.

    Args:
        image: The input image to resize
        target_size: The target size for the square output

    Returns:
        A square image with dimensions (target_size, target_size)
    """
    width, height = image.size

    # First, resize the image so that the smaller dimension equals target_size
    if width < height:
        # Portrait - width is smaller
        ratio = target_size / width
        new_width = target_size
        new_height = int(height * ratio)
    else:
        # Landscape - height is smaller
        ratio = target_size / height
        new_height = target_size
        new_width = int(width * ratio)

    # Resize the image
    resized_img = image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate crop box for center crop
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    # Crop to square
    return resized_img.crop((left, top, right, bottom))


def load_images(image_urls: List[str], size: int, cache_dir: str = "images") -> torch.Tensor:
    """
    Load images from URLs, resize them to squares, and convert to tensors.
    Uses caching to avoid re-downloading.

    Args:
        image_urls: List of URLs to download images from
        size: Target size for the square images
        cache_dir: Directory to store cached images

    Returns:
        A batch of image tensors with shape [batch_size, channels, height, width]
    """
    imgs = []
    for url in image_urls:
        # Get filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # Create directory for downloads
        image_dir = Path(cache_dir) / filename
        original_path = image_dir / f"{filename}_original.jpg"
        resized_path = image_dir / f"{filename}_{size}.jpg"

        # Load or download images
        if not resized_path.exists():
            image_dir.mkdir(parents=True, exist_ok=True)
            if not original_path.exists():
                download_image(url, original_path)
            img_original = Image.open(original_path)
            img_resized = resize_and_crop_square(img_original, size)
            img_resized.save(resized_path)

        img_resized = Image.open(resized_path)
        img_tensor = transforms.ToTensor()(img_resized)
        imgs.append(img_tensor)

    return torch.stack(imgs)


def display_images(*images: Tuple[str, Image.Image]):
    """
    Display an arbitrary number of images side by side.
    The image name will be displayed in the title.
    """
    num_images = len(images)
    if num_images == 0:
        raise ValueError("At least one image must be provided.")

    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 7))

    # Ensure axes is always iterable
    axes: list[Axes] = axes if num_images > 1 else [axes]

    for ax, (name, img) in zip(axes, images):
        ax.imshow(np.array(img))
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def load_flux_vae(device: torch.device) -> AutoencoderKL:
    """
    Load the VAE component from the FLUX.1-dev model.

    Args:
        device: The torch device to load the model on (CPU or GPU)

    Returns:
        The loaded VAE model
    """
    # Load the VAE directly from the specified path
    snapshot_path = Path(huggingface_hub.snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        allow_patterns=["vae/*"],
        local_dir="."))

    vae_path = snapshot_path / "vae"
    vae = AutoencoderKL.from_pretrained(vae_path, use_safetensors=True)
    return vae.to(device)


def plot_tensor(t: torch.Tensor, mode: str = "image", title: Optional[str] = None) -> None:
    """
    Helper function to plot a tensor based on the specified mode.

    Args:
        t: Tensor to plot
        mode: Plotting mode - 'image' for RGB images, 'single' for individual channels, 
              'raps' for 1D radial plots
        title: Optional title for the plot

    Raises:
        ValueError: If the tensor shape is incompatible with the specified mode
    """
    # Ensure tensor is on CPU
    t = t.cpu()

    if title:
        plt.suptitle(title, fontsize=16)

    if mode == 'image':
        # RGB images with shape (batch, 3, height, width)
        if len(t.shape) == 4 and t.shape[1] == 3:
            b, _, h, w = t.shape
            for idx in range(b):
                # Convert to HWC format for RGB images
                image = t[idx].permute(1, 2, 0)
                plt.imshow(image)
                plt.axis('off')
                plt.title(f"RGB Image {idx + 1}")
                plt.show()
        else:
            raise ValueError(
                f"Unsupported tensor shape {t.shape} for image mode. Expected (b, 3, h, w).")

    elif mode == 'single':
        # Multi-channel tensors with shape (batch, channels, height, width)
        if len(t.shape) == 4 and t.shape[1] > 0:
            b, c, h, w = t.shape
            for idx in range(b):
                # Calculate rows needed for a 4-column grid
                rows = max(1, (c + 3) // 4)
                fig, axes = plt.subplots(
                    rows, min(c, 4), figsize=(16, 4 * rows))
                # Ensure axes is flat and iterable
                axes = np.array(axes).reshape(-1) if c > 1 else [axes]

                # Plot each channel as a single-channel image
                for i in range(c):
                    if i < len(axes):
                        ax = axes[i]
                        ax.imshow(t[idx, i], cmap='viridis')
                        ax.axis('off')
                        ax.set_title(f"Channel {i + 1}")

                plt.suptitle(f"Image {idx + 1}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()
        else:
            raise ValueError(
                f"Unsupported tensor shape {t.shape} for single mode. Expected (b, c, h, w).")

    elif mode == 'raps':
        # 1D data with shape (batch, channels, data_points)
        if len(t.shape) == 3 and t.shape[1] > 0:
            b, c, x = t.shape
            for idx in range(b):
                fig, axes = plt.subplots(c, 1, figsize=(8, 2 * c))
                # Make it iterable if only one channel
                axes = [axes] if c == 1 else axes

                # Plot each channel as a 1D line plot
                for i in range(c):
                    ax = axes[i]
                    ax.plot(t[idx, i])
                    ax.set_title(f"Channel {i + 1}")
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Value")

                plt.suptitle(f"1D Data for Image {idx + 1}", fontsize=16)
                plt.tight_layout()
                plt.show()
        else:
            raise ValueError(
                f"Unsupported tensor shape {t.shape} for raps mode. Expected (b, c, x).")

    else:
        raise ValueError("Mode must be 'image', 'single', or 'raps'.")


def plot_fft(t: torch.Tensor, batch: Union[Tuple[int], int, None]):
    mag = torch.abs(t)
    phase = torch.angle(t)
    batch_size, channels, height, width = t.shape

    plt.figure(figsize=(20, 6))

    for b in range(batch_size):
        if isinstance(batch, Iterable):
            if b not in batch:
                continue
        elif isinstance(batch, int):
            if b != batch:
                continue
        elif batch is None:
            pass

          # Plot all channels for magnitude
        for c in range(channels):
            plt.subplot(2, channels, c + 1)
            plt.imshow(torch.log(torch.clamp(
                mag[b, c], min=1e-10)).cpu(), cmap='gray')
            plt.title(f"Mag-Img{b+1}Ch{c+1}")
            plt.axis('off')

        # Plot all channels for phase
        for c in range(channels):
            plt.subplot(2, channels, channels + c + 1)
            plt.imshow(phase[b, c].cpu(), cmap='twilight')
            plt.title(f"Ph-Img{b+1}Ch{c+1}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def weighted_mean(values, weights):
    """Calculate weighted mean of values with weights."""
    values, weights = np.asarray(values), np.asarray(weights)
    return np.sum(values * weights) / np.sum(weights)


def calc_raps(fft_shifted: torch.Tensor) -> torch.Tensor:
    """
    Calculate Radially Averaged Power Spectrum (RAPS) from FFT data.

    Args:
        fft_shifted: Shifted FFT data with DC component at center

    Returns:
        Tensor containing RAPS with shape (batch, channels, max_radius)
    """
    # Compute the power spectrum (magnitude squared)
    # Paseval's Theorem and sampling rate norm
    power_spectrum = torch.abs(fft_shifted) ** 2 / fft_shifted.size(dim=2) ** 2

    # Get tensor dimensions
    batch_size, channels, height, width = power_spectrum.shape

    # Calculate center of the image (where low frequencies are located)
    center_x, center_y = width // 2, height // 2

    # Create a grid of distance from the center
    y, x = np.indices((height, width), dtype=np.float64)
    x = x - center_x
    y = y - center_y
    radius = np.sqrt(x ** 2 + y ** 2)

    # Find the maximum radius
    smaller_dim = min(width, height)
    max_radius = smaller_dim // 2 + (1 if smaller_dim % 2 else 0)

    # Process batch data
    batch_results = []
    for b in range(batch_size):
        channel_results = []
        for c in range(channels):
            # Initialize bins for radial averaging
            radial_samples = [[] for _ in range(max_radius)]
            radial_weights = [[] for _ in range(max_radius)]

            # Gather samples in radial bins with interpolation
            for i in range(height):
                for j in range(width):
                    r = radius[i, j]
                    lower_bin = np.floor(r).astype(int)
                    upper_bin = np.ceil(r).astype(int)
                    delta = upper_bin - r

                    # Assign power to lower bin with appropriate weight
                    if lower_bin in range(max_radius):
                        radial_samples[lower_bin].append(
                            power_spectrum[b, c, i, j].item())
                        radial_weights[lower_bin].append(delta)

                    # Assign power to upper bin with appropriate weight
                    if upper_bin in range(max_radius):
                        radial_samples[upper_bin].append(
                            power_spectrum[b, c, i, j].item())
                        radial_weights[upper_bin].append(1 - delta)

            # Calculate weighted means for each radial bin
            radial_result = torch.empty(len(radial_samples))
            for bin, (samples, weights) in enumerate(zip(radial_samples, radial_weights)):
                if samples:  # Only calculate if there are samples in this bin
                    radial_result[bin] = weighted_mean(samples, weights)
                else:
                    radial_result[bin] = 0.0

            channel_results.append(radial_result)

        batch_result = torch.stack(channel_results)
        batch_results.append(batch_result)

    return torch.stack(batch_results)


def fit_loglog_polynomial(f, P, degree=2):
    """
    Fit a 2nd-degree polynomial in log-log space: log10(P) = a (log10(f))^2 + b log10(f) + c
    
    Parameters:
    f : torch.Tensor or np.ndarray, frequency values (positive)
    P : torch.Tensor or np.ndarray, RAPS values (positive)
    degree : int, polynomial degree (default=2 for quadratic)
    
    Returns:
    coefficients : np.ndarray, polynomial coefficients [a, b, c] for a x^2 + b x + c
    P_fit : np.ndarray, fitted values in original space
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()
    
    # Remove non-positive values to avoid log issues
    valid = (f > 0) & (P > 0)
    f = f[valid]
    P = P[valid]
    
    if len(f) == 0:
        raise ValueError("No valid data points (f and P must be positive).")
    
    # Transform to log-space
    log_f = np.log10(f)
    log_P = np.log10(P)
    
    # Fit a 2nd-degree polynomial in log-log space
    coefficients = np.polyfit(log_f, log_P, deg=degree)
    
    # Compute fitted values in log-space: log10(P_fit) = a (log10(f))^2 + b log10(f) + c
    log_P_fit = np.polyval(coefficients, log_f)
    
    # Transform back to original space: P_fit = 10^(log_P_fit)
    P_fit = 10**log_P_fit
    
    return coefficients, P_fit

if __name__ == "__main__":
    print(f"Testing RAPS calculation on random tensor...")
    test_tensor = torch.rand((2, 3, 64, 64), dtype=torch.complex64)
    result = calc_raps(test_tensor)
    print(f"Input shape: {test_tensor.shape}, Output shape: {result.shape}")
