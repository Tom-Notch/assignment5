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

from models import cls_model
from utils import create_dir
from utils import sample_random_rotation_matrices
from utils import viz_cls


def create_parser() -> argparse.ArgumentParser:
    """Creates a parser for command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_cls_class", type=int, default=3, help="The number of classes"
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
        help="Number of random samples to visualize",
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

    parser.add_argument("--test_data", type=str, default="./data/cls/data_test.npy")
    parser.add_argument("--test_label", type=str, default="./data/cls/label_test.npy")
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
    create_dir(os.path.join(args.output_dir, "cls_visualizations"))

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class)

    # Load Model Checkpoint
    model_path = f"./checkpoints/cls/{args.load_checkpoint}.pt"
    with open(model_path, "rb") as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print(f"successfully loaded checkpoint from {model_path}")

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:, ind, :])
    test_label = torch.from_numpy(np.load(args.test_label))

    if args.random_rotation:
        rotation_matrices = sample_random_rotation_matrices(test_data.shape[0])
        test_data = (rotation_matrices @ test_data.mT).mT

    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data).argmax(dim=-1)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (
        test_label.size()[0]
    )
    print(f"test accuracy: {test_accuracy}")

    # Class names for visualization
    class_names: Dict[int, str] = {0: "Chair", 1: "Vase", 2: "Lamp"}

    # Visualize successful predictions for each class
    print("\nVisualizing successful predictions...")
    for true_class in range(args.num_cls_class):
        # Find indices where prediction matches true class
        mask = (test_label == true_class) & (pred_label == true_class)
        if mask.any():
            success_idx = torch.where(mask)[0][0].item()
            points = test_data[success_idx].to(args.device)
            class_name = class_names[true_class]
            title = f"Success Case - {class_name}"
            save_path = os.path.join(
                args.output_dir, "cls_visualizations", f"success_{class_name}.gif"
            )
            viz_cls(points, save_path, args.device, title)
            print(f"Saved success case for {class_name}")
        else:
            print(f"No success cases found for {class_names[true_class]}")

    # Visualize failure cases for each class
    print("\nVisualizing failure cases...")
    for true_class in range(args.num_cls_class):
        # Find indices where true class is correct but prediction is wrong
        mask = (test_label == true_class) & (pred_label != true_class)
        if mask.any():
            failure_idx = torch.where(mask)[0][0].item()
            points = test_data[failure_idx].to(args.device)
            true_class_name = class_names[true_class]
            pred_class_name = class_names[pred_label[failure_idx].item()]
            title = (
                f"Failure Case - GT: {true_class_name}, Predicted: {pred_class_name}"
            )
            save_path = os.path.join(
                args.output_dir, "cls_visualizations", f"failure_{true_class_name}.gif"
            )
            viz_cls(points, save_path, args.device, title)
            print(f"Saved failure case for {true_class_name}")
        else:
            print(f"No failure cases found for {class_names[true_class]}")
