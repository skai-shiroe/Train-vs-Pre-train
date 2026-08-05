"""
Module d'attention multi-tête.

Ce module contient l'implémentation de l'attention multi-tête
pour le modèle Transformer.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Attention multi-tête pour le Transformer.
    
    Implémentation de l'attention multi-tête avec projections
    linéaires et softmax.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialiser l'attention multi-tête.
        
        Args:
            d_model: Dimension du modèle
            num_heads: Nombre de têtes d'attention
            dropout: Taux de dropout
            
        TODO: Implémenter l'initialisation
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # TODO: Créer les projections linéaires Q, K, V et la projection de sortie
        # self.w_q = nn.Linear(d_model, d_model)
        # self.w_k = nn.Linear(d_model, d_model)
        # self.w_v = nn.Linear(d_model, d_model)
        # self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass de l'attention multi-tête.
        
        Args:
            query: Tensor de requêtes (batch_size, seq_len, d_model)
            key: Tensor de clés (batch_size, seq_len, d_model)
            value: Tensor de valeurs (batch_size, seq_len, d_model)
            mask: Masque d'attention optionnel
            
        Returns:
            Tensor de sortie (batch_size, seq_len, d_model)
            
        TODO: Implémenter le forward pass
        """
        # 1. Projeter Q, K, V
        # 2. Reshaper pour les têtes multiples
        # 3. Calculer les scores d'attention
        # 4. Appliquer le masque si fourni
        # 5. Softmax et dropout
        # 6. Combiner les têtes
        # 7. Projection finale
        pass