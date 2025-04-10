#!/usr/bin/env python3
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import imageio
import numpy as np
import pytorch3d
import torch
from pytorch3d.renderer import AlphaCompositor
from pytorch3d.renderer import PointsRasterizationSettings
from pytorch3d.renderer import PointsRasterizer
from pytorch3d.renderer import PointsRenderer
from torch import nn


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    args: Any,
    best: bool = False,
) -> None:
    if best:
        path = os.path.join(args.checkpoint_dir, "best_model.pt")
    else:
        path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), path)


def create_dir(directory: str) -> None:
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_points_renderer(
    image_size: int = 256,
    device: Optional[torch.device] = None,
    radius: float = 0.01,
    background_color: Tuple[float, float, float] = (1, 1, 1),
) -> PointsRenderer:
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.

    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg(
    verts: torch.Tensor,
    labels: torch.Tensor,
    path: str,
    device: torch.device,
) -> None:
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size = 256
    background_color: Tuple[float, float, float] = (1, 1, 1)
    colors: List[List[float]] = [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim: List[float] = [180 - 12 * i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim, device=device
    )
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    # Get the actual number of points
    num_points = verts.shape[0]

    # Create tensors with the correct number of points
    sample_verts = verts.unsqueeze(0).repeat(30, 1, 1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1, num_points, 3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels == i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30, 1, 1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(
        points=sample_verts, features=sample_colors
    ).to(device)

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color, device=device
    )
    rend = renderer(point_cloud, cameras=c).cpu().numpy()  # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15, loop=0)


def viz_cls(
    verts: torch.Tensor,
    path: str,
    device: torch.device,
    title: Optional[str] = None,
) -> None:
    """
    visualize classification result
    output: a 360-degree gif
    """
    image_size = 256
    background_color: Tuple[float, float, float] = (1, 1, 1)
    color: List[float] = [0.7, 0.7, 1.0]  # Light blue color for all points

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim: List[float] = [180 - 12 * i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim, device=device
    )
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30, 1, 1).to(torch.float).to(device)
    sample_colors = (
        torch.tensor(color, device=device).unsqueeze(0).repeat(1, verts.shape[0], 1)
    )
    sample_colors = sample_colors.repeat(30, 1, 1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(
        points=sample_verts, features=sample_colors
    ).to(device)

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color, device=device
    )
    rend = renderer(point_cloud, cameras=c).cpu().numpy()  # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    # Add title if provided
    if title:
        from PIL import Image, ImageDraw, ImageFont

        frames: List[np.ndarray] = []
        for frame in rend:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            # You might need to adjust font size and position based on your needs
            draw.text((10, 10), title, fill=(0, 0, 0))
            frames.append(np.array(img))
        rend = np.array(frames)

    imageio.mimsave(path, rend, fps=15, loop=0)


def sample_random_rotation_matrices(batch_size: int = 1) -> torch.Tensor:
    """
    Generate 'batch_size' random rotation matrices in SO(3) using quaternions.
    True randomness is mathematically guaranteed.

    Args:
        batch_size (int): Number of random rotation matrices to generate.

    Returns:
        torch.Tensor of shape (batch_size, 3, 3) with each sub-tensor an orthonormal rotation matrix.
    """
    # Step 1: sample (batch_size x 4) standard normal random values
    q = torch.randn(batch_size, 4)

    # Step 2: normalize to get unit quaternions
    q = q / q.norm(dim=1, keepdim=True)

    # Separate out components for readability
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    # Step 3: Convert each quaternion to a 3×3 rotation matrix
    # Shape will be (batch_size, 3, 3)
    R = torch.empty((batch_size, 3, 3))

    R[:, 0, 0] = 1 - 2 * (q2**2 + q3**2)
    R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)

    R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[:, 1, 1] = 1 - 2 * (q1**2 + q3**2)
    R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)

    R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[:, 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return R
