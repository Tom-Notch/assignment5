#!/usr/bin/env python3
import argparse
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn
from torch import Tensor

from data_loader import get_data_loader
from models import seg_model
from utils import create_dir
from utils import sample_random_rotation_matrices
from utils import viz_seg


def create_parser() -> argparse.ArgumentParser:
    """Creates a parser for command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_seg_class",
        type=int,
        default=6,
        help="The number of segmentation classes",
    )
    parser.add_argument(
        "--num_points",
        "-n",
        type=int,
        default=10000,
        help="The number of points per object to be included in the input data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--random_rotation",
        "-r",
        action="store_true",
        help="Whether to use random rotation",
    )

    # Directories and checkpoint/sample iterations
    parser.add_argument("--load_checkpoint", type=str, default="best_model")
    parser.add_argument(
        "--i", type=int, default=0, help="index of the object to visualize"
    )

    parser.add_argument("--test_data", type=str, default="./data/seg/data_test.npy")
    parser.add_argument("--test_label", type=str, default="./data/seg/label_test.npy")
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument(
        "--exp_name", type=str, default="exp", help="The name of the experiment"
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    create_dir(args.output_dir)
    create_dir(os.path.join(args.output_dir, "seg_visualizations"))

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class)

    # Load Model Checkpoint
    model_path = f"./checkpoints/seg/{args.load_checkpoint}.pt"
    with open(model_path, "rb") as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print(f"successfully loaded checkpoint from {model_path}")

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:, ind, :])
    test_label = torch.from_numpy((np.load(args.test_label))[:, ind])

    if args.random_rotation:
        rotation_matrices = sample_random_rotation_matrices(test_data.shape[0])
        test_data = (rotation_matrices @ test_data.mT).mT

    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data).argmax(dim=-1)

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (
        test_label.reshape((-1, 1)).size()[0]
    )
    print(f"test accuracy: {test_accuracy}")

    # Calculate per-sample accuracy
    sample_accuracies = []
    for i in range(len(test_data)):
        accuracy = (pred_label[i] == test_label[i]).float().mean().item()
        sample_accuracies.append((i, accuracy))

    # Sort samples by accuracy
    sample_accuracies.sort(key=lambda x: x[1])

    # Visualize best predictions
    print("\nVisualizing best predictions...")
    for i in range(min(3, len(sample_accuracies))):
        idx, accuracy = sample_accuracies[-(i + 1)]
        print(f"Sample {idx}: Accuracy = {accuracy:.4f}")
        # Visualize ground truth
        viz_seg(
            test_data[idx],
            test_label[idx],
            f"{args.output_dir}/seg_visualizations/gt_best_{i}.gif",
            args.device,
        )
        # Visualize prediction
        viz_seg(
            test_data[idx],
            pred_label[idx],
            f"{args.output_dir}/seg_visualizations/prediction_best_{i}.gif",
            args.device,
        )

    # Visualize worst predictions
    print("\nVisualizing worst predictions...")
    for i in range(min(2, len(sample_accuracies))):
        idx, accuracy = sample_accuracies[i]
        print(f"Sample {idx}: Accuracy = {accuracy:.4f}")
        # Visualize ground truth
        viz_seg(
            test_data[idx],
            test_label[idx],
            f"{args.output_dir}/seg_visualizations/gt_worst_{i}.gif",
            args.device,
        )
        # Visualize prediction
        viz_seg(
            test_data[idx],
            pred_label[idx],
            f"{args.output_dir}/seg_visualizations/prediction_worst_{i}.gif",
            args.device,
        )
