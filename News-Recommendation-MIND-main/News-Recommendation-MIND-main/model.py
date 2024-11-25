
"""

This module defines the Matrix Factorization model for the news recommendation system
using PyTorch Lightning.

Classes:
    NewsMF: A PyTorch Lightning module implementing Matrix Factorization for recommendation.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class NewsMF(pl.LightningModule):
    """
    A PyTorch Lightning module for Matrix Factorization based news recommendation.

    Attributes:
        dim (int): Dimensionality of the embedding vectors.
        useremb (nn.Embedding): Embedding layer for users.
        itememb (nn.Embedding): Embedding layer for items.

    Methods:
        forward(user, item): Computes the interaction score between a user and an item.
        training_step(batch, batch_idx): Defines the training step.
        validation_step(batch, batch_idx): Defines the validation step.
        configure_optimizers(): Configures the optimizer.
    """
    def __init__(self, num_users, num_items, dim=10):
        """
        Initializes the NewsMF model with user and item embedding layers.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items (news articles).
            dim (int, optional): Dimensionality of the embedding vectors. Defaults to 10.
        """
        super().__init__()
        self.dim = dim
        self.useremb = nn.Embedding(num_embeddings=num_users, embedding_dim=dim)
        self.itememb = nn.Embedding(num_embeddings=num_items, embedding_dim=dim)

    def forward(self, user, item):
        """
        Computes the interaction score between a user and an item.

        Args:
            user (torch.Tensor): Tensor containing user indices.
            item (torch.Tensor): Tensor containing item indices.

        Returns:
            torch.Tensor: Tensor containing interaction scores.
        """
        uservec = self.useremb(user)
        itemvec = self.itememb(item)
        score = (uservec * itemvec).sum(-1).unsqueeze(-1)
        return score

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (dict): Batch of data containing user indices, clicked item indices, and non-clicked item indices.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss.
        """
        score_click = self.forward(batch['userIdx'], batch['click'])
        score_noclick = self.forward(batch['userIdx'], batch['noclick'])
        scores_all = torch.concat((score_click, score_noclick), dim=1)
        loss = nn.CrossEntropyLoss()(scores_all, torch.zeros(batch['userIdx'].size(0), device=scores_all.device).long())
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (dict): Batch of data containing user indices, clicked item indices, and non-clicked item indices.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)
