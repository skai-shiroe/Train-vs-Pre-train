"""
Module du décodeur du Transformer.

Ce module contient l'implémentation du décodeur du Transformer,
qui génère les résumés en utilisant l'attention sur l'encodeur.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DecoderLayer(nn.Module):
    """
    Une couche du décodeur du Transformer.
    
    Contient une couche d'attention masquée, une couche d'attention
    cross-attention sur l'encodeur, et une couche feed-forward.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialiser une couche de décodeur.
        
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
        # - MultiHeadAttention (self-attention masquée)
        # - MultiHeadAttention (cross-attention avec encodeur)
        # - FeedForward
        # - LayerNorm (x3)
        # - Dropout (x3)
        
    def forward(self, x: torch.Tensor, 
                encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass d'une couche de décodeur.
        
        Args:
            x: Tensor d'entrée du décodeur (batch_size, seq_len, d_model)
            encoder_output: Sortie de l'encodeur (batch_size, seq_len, d_model)
            src_mask: Masque pour l'attention sur l'encodeur
            tgt_mask: Masque pour le self-attention du décodeur
            
        Returns:
            Tensor de sortie (batch_size, seq_len, d_model)
            
        TODO: Implémenter le forward pass
        """
        # 1. Self-attention masquée avec connexion résiduelle
        # 2. Cross-attention sur l'encodeur avec connexion résiduelle
        # 3. Feed-forward avec connexion résiduelle
        # 4. Normalisations
        pass


class Decoder(nn.Module):
    """
    Décodeur complet du Transformer.
    
    Empile N couches de décodeur pour générer les séquences de sortie.
    """
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float = 0.1):
        """
        Initialiser le décodeur.
        
        Args:
            num_layers: Nombre de couches de décodeur
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
        
        # TODO: Créer la pile de couches de décodeur
        # self.layers = nn.ModuleList([
        #     DecoderLayer(d_model, num_heads, d_ff, dropout)
        #     for _ in range(num_layers)
        # ])
        # self.norm = LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass du décodeur complet.
        
        Args:
            x: Tensor d'entrée du décodeur (batch_size, seq_len, d_model)
            encoder_output: Sortie de l'encodeur (batch_size, seq_len, d_model)
            src_mask: Masque pour l'attention sur l'encodeur
            tgt_mask: Masque pour le self-attention du décodeur
            
        Returns:
            Tensor de sortie (batch_size, seq_len, d_model)
            
        TODO: Implémenter le forward pass
        """
        # Passer par chaque couche de décodeur
        # Normalisation finale
        pass