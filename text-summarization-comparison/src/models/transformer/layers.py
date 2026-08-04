"""
Couches du Transformer.

Ce module contient les implémentations des couches feed-forward,
normalisation et autres composants de base du Transformer.
"""

from typing import Optional
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
            d_ff: Dimension de la couche feed-forward
            dropout: Taux de dropout
            
        TODO: Implémenter l'initialisation
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
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


class LayerNorm(nn.Module):
    """
    Normalisation de couche.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialiser la normalisation de couche.
        
        Args:
            d_model: Dimension du modèle
            eps: Epsilon pour la stabilité numérique
            
        TODO: Implémenter l'initialisation
        """
        super().__init__()
        # self.gamma = nn.Parameter(torch.ones(d_model))
        # self.beta = nn.Parameter(torch.zeros(d_model))
        # self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Appliquer la normalisation de couche.
        
        Args:
            x: Tensor d'entrée (batch_size, seq_len, d_model)
            
        Returns:
            Tensor normalisé
            
        TODO: Implémenter le forward pass
        """
        pass


class ResidualConnection(nn.Module):
    """
    Connexion résiduelle avec normalisation.
    
    Architecture: x -> Sublayer -> Dropout -> Add -> Norm
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initialiser la connexion résiduelle.
        
        Args:
            d_model: Dimension du modèle
            dropout: Taux de dropout
            
        TODO: Implémenter l'initialisation
        """
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Appliquer la connexion résiduelle.
        
        Args:
            x: Tensor d'entrée
            sublayer: Couche à appliquer (attention ou feed-forward)
            
        Returns:
            Tensor après connexion résiduelle
            
        TODO: Implémenter le forward pass
        """
        # return self.norm(x + self.dropout(sublayer(x)))
        pass