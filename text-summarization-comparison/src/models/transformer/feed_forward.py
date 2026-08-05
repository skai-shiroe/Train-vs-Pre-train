"""
Module de couche feed-forward.

Ce module contient l'implémentation de la couche feed-forward
utilisée dans l'encodeur et le décodeur du Transformer.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """
    Couche feed-forward du Transformer.
    
    Architecture: Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialiser la couche feed-forward.
        
        Args:
            d_model: Dimension du modèle
            d_ff: Dimension de la couche feed-forward (généralement 4 * d_model)
            dropout: Taux de dropout
            
        TODO: Implémenter l'initialisation
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # TODO: Créer les couches linéaires
        # self.linear1 = nn.Linear(d_model, d_ff)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la couche feed-forward.
        
        Args:
            x: Tensor d'entrée (batch_size, seq_len, d_model)
            
        Returns:
            Tensor de sortie (batch_size, seq_len, d_model)
            
        TODO: Implémenter le forward pass
        """
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        pass