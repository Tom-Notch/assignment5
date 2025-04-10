#!/usr/bin/env python3
import argparse
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_data_loader
from models import cls_model
from models import seg_model
from utils import create_dir
from utils import save_checkpoint


def train(
    train_dataloader: DataLoader,
    model: nn.Module,
    opt: optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    writer: SummaryWriter,
) -> Tensor:

    model.train()
    step: int = epoch * len(train_dataloader)
    epoch_loss: Tensor = torch.tensor(0.0, device=args.device)

    for i, batch in enumerate(train_dataloader):
        point_clouds: Tensor
        labels: Tensor
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)

        # ------ TO DO: Forward Pass ------
        predictions: Tensor = model(point_clouds)

        if args.task == "seg":
            labels = labels.reshape([-1])
            predictions = predictions.reshape([-1, args.num_seg_class])

        # Compute Loss
        criterion: nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        loss: Tensor = criterion(predictions, labels)
        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("train_loss", loss.item(), step + i)

    return epoch_loss


def test(
    test_dataloader: DataLoader,
    model: nn.Module,
    epoch: int,
    args: argparse.Namespace,
    writer: SummaryWriter,
) -> float:

    model.eval()

    # Evaluation in Classification Task
    if args.task == "cls":
        correct_obj: int = 0
        num_obj: int = 0
        for batch in test_dataloader:
            point_clouds: Tensor
            labels: Tensor
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                pred_labels: Tensor = model(point_clouds).argmax(dim=-1)
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

        # Compute Accuracy of Test Dataset
        accuracy: float = correct_obj / num_obj

    # Evaluation in Segmentation Task
    else:
        correct_point: int = 0
        num_point: int = 0
        for batch in test_dataloader:
            point_clouds: Tensor
            labels: Tensor
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                pred_labels: Tensor = model(point_clouds).argmax(dim=-1)

            correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            num_point += labels.view([-1, 1]).size()[0]

        # Compute Accuracy of Test Dataset
        accuracy: float = correct_point / num_point

    writer.add_scalar("test_acc", accuracy, epoch)
    return accuracy


def main(args: argparse.Namespace) -> None:
    """Loads the data, creates checkpoint and sample directories, and starts the training loop."""

    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir("./logs")

    # Tensorboard Logger
    writer: SummaryWriter = SummaryWriter(f"./logs/{args.task}_{args.exp_name}")

    # ------ TO DO: Initialize Model ------
    model: nn.Module
    if args.task == "cls":
        model = cls_model(num_classes=3).to(args.device)
    else:
        model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)

    # Load Checkpoint
    if args.load_checkpoint:
        model_path: str = f"{args.checkpoint_dir}/{args.load_checkpoint}.pt"
        with open(model_path, "rb") as f:
            state_dict: Dict[str, Any] = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        print(f"successfully loaded checkpoint from {model_path}")

    # Optimizer
    opt: optim.Adam = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # Dataloader for Training & Testing
    train_dataloader: DataLoader = get_data_loader(args=args, train=True)
    test_dataloader: DataLoader = get_data_loader(args=args, train=False)

    print("successfully loaded data")

    best_acc: float = -1.0

    print(f"======== start training for {args.task} task ========")
    print(
        f"(check tensorboard for plots of experiment logs/{args.task}_{args.exp_name})"
    )

    for epoch in range(args.num_epochs):
        # Train
        train_epoch_loss: Tensor = train(
            train_dataloader, model, opt, epoch, args, writer
        )

        # Test
        current_acc: float = test(test_dataloader, model, epoch, args, writer)

        print(
            f"epoch: {epoch}   train loss: {train_epoch_loss.item():.4f}   test accuracy: {current_acc:.4f}"
        )

        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print(f"checkpoint saved at epoch {epoch}")
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # Save Best Model Checkpoint
        if current_acc >= best_acc:
            best_acc = current_acc
            print(f"best model saved at epoch {epoch}")
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)

    print("======== training completes ========")


def create_parser() -> argparse.ArgumentParser:
    """Creates a parser for command-line arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument("--task", type=str, default="cls", help="The task: cls or seg")
    parser.add_argument(
        "--num_seg_class",
        type=int,
        default=6,
        help="The number of segmentation classes",
    )

    # Training hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The number of images in a batch."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of threads to use for the DataLoader.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The learning rate (default 0.001)"
    )

    parser.add_argument(
        "--exp_name", type=str, default="exp", help="The name of the experiment"
    )

    # Directories and checkpoint/sample iterations
    parser.add_argument("--main_dir", type=str, default="./data/")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=10)

    parser.add_argument("--load_checkpoint", type=str, default="")

    return parser


if __name__ == "__main__":
    parser: argparse.ArgumentParser = create_parser()
    args: argparse.Namespace = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir = (
        args.checkpoint_dir + "/" + args.task
    )  # checkpoint directory is task specific

    main(args)
