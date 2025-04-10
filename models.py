#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

from point_transformer import PointTransformer


# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Point Transformer backbone
        self.backbone = PointTransformer(
            in_dim=3, out_dim=256, num_blocks=4, num_neighbors=16  # xyz coordinates
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: tensor of size (B, N, 3), where B is batch size and N is the number of points per object
        output: tensor of size (B, num_classes)
        """
        # Extract features using Point Transformer
        features = self.backbone(points, points)  # (B, N, 1024)

        # Global max pooling
        global_features = features.max(dim=1)[0]  # (B, 1024)

        # Classification
        classes = self.classifier(global_features)
        return classes


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes: int = 6):
        super().__init__()

        # Point Transformer backbone
        self.backbone = PointTransformer(
            in_dim=3, out_dim=256, num_blocks=4, num_neighbors=16  # xyz coordinates
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, num_seg_classes, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: tensor of size (B, N, 3), where B is batch size and N is the number of points per object
        output: tensor of size (B, N, num_seg_classes)
        """
        # Extract features using Point Transformer
        features = self.backbone(points, points)  # (B, N, 1024)

        # Segmentation
        semantics = self.seg_head(features.transpose(1, 2)).transpose(1, 2)
        return semantics
