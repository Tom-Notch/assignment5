#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.feature_extractor_stage_1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.feature_extractor_stage_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: tensor of size (B, N, 3), where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        """
        points_transposed = points.transpose(2, 1)
        features = self.feature_extractor_stage_2(
            self.feature_extractor_stage_1(points_transposed)
        )
        classes = self.classifier(features.max(dim=-1)[0])
        return classes


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes: int = 6):
        super().__init__()
        self.feature_extractor_stage_1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.feature_extractor_stage_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(1088, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, num_seg_classes, kernel_size=1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: tensor of size (B, N, 3), where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        """
        points_transposed = points.transpose(2, 1)

        features_stage_1 = self.feature_extractor_stage_1(points_transposed)
        features_stage_2 = self.feature_extractor_stage_2(features_stage_1)
        global_feature = features_stage_2.max(dim=-1, keepdim=True)[0].repeat(
            1, 1, features_stage_1.shape[-1]
        )

        semantics = self.classifier(
            torch.cat([features_stage_1, global_feature], dim=1)
        )

        return semantics.transpose(2, 1)
