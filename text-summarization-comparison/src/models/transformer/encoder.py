"""
Module de l'encodeur du Transformer.

Ce module contient l'implémentation de l'encodeur du Transformer,
qui traite les séquences d'entrée (articles) via des couches d'attention.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EncoderLayer(nn.Module):
    """
    Une couche de l'encodeur du Transformer.
    
    Contient une couche d'attention multi-tête et une couche feed-forward,
    avec des connexions résiduelles et une normalisation.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialiser une couche d'encodeur.
        
        Args:
            d_model: Dimension du modèle
            num_heads: Nombre de têtes d'attention
            d_ff: Dimension de la couche feed-forward
            dropout: Taux de dropout
            
        TODO: Implémenter l'initialisation
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # TODO: Créer les composants
        # - MultiHeadAttention
        # - FeedForward
        # - LayerNorm (x2)
        # - Dropout (x2)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass d'une couche d'encodeur.
        
        Args:
            x: Tensor d'entrée (batch_size, seq_len, d_model)
            mask: Masque d'attention optionnel
            
        Returns:
            Tensor de sortie (batch_size, seq_len, d_model)
            
        TODO: Implémenter le forward pass
        """
        # 1. Self-attention avec connexion résiduelle
        # 2. Normalisation
        # 3. Feed-forward avec connexion résiduelle
        # 4. Normalisation finale
        pass


class Encoder(nn.Module):
    """
    Encodeur complet du Transformer.
    
    Empile N couches d'encodeur pour traiter les séquences d'entrée.
    """
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        """
        Initialiser l'encodeur.
        
        Args:
            num_layers: Nombre de couches d'encodeur
            d_model: Dimension du modèle
            num_heads: Nombre de têtes d'attention
            d_ff: Dimension de la couche feed-forward
            dropout: Taux de dropout
            
        TODO: Implémenter l'initialisation
        """
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # TODO: Créer la pile de couches d'encodeur
        # self.layers = nn.ModuleList([
        #     EncoderLayer(d_model, num_heads, d_ff, dropout)
        #     for _ in range(num_layers)
        # ])
        # self.norm = LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass de l'encodeur complet.
        
        Args:
            x: Tensor d'entrée (batch_size, seq_len, d_model)
            mask: Masque d'attention optionnel
            
        Returns:
            Tensor de sortie (batch_size, seq_len, d_model)
            
        TODO: Implémenter le forward pass
        """
        # Passer par chaque couche d'encodeur
        # Normalisation finale
        pass