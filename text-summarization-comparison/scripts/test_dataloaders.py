#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test des DataLoaders PyTorch.
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_dataloader

# Chemin vers les donnes
BASE_PATH = Path(__file__).parent.parent / "notebooks" / "data" / "processed"

print("=" * 70)
print(" TEST DATALOADER: TRANSFORMER | train_10")
print("=" * 70)

# Creer le DataLoader
dataloader = create_dataloader(
    data_path=BASE_PATH,
    split="transformer/train_10",
    model_type="transformer",
    batch_size=8,
    shuffle=True,
    num_workers=0
)

print(f"\nDataLoader cree: {len(dataloader)} batches")
print(f"\n1. INFORMATIONS:")
print(f"   Split: transformer/train_10")
print(f"   Modele: transformer")
print(f"   Batch size: 8")
print(f"   Nombre de batches: {len(dataloader)}")

print(f"\n2. TEST DE CHARGEMENT D'UN BATCH:")
batch = next(iter(dataloader))

print(f"   • input_ids shape: {batch['input_ids'].shape}")
print(f"   • attention_mask shape: {batch['attention_mask'].shape}")
print(f"   • labels shape: {batch['labels'].shape}")

print(f"\n3. VERIFICATION DES TYPES:")
print(f"   • input_ids dtype: {batch['input_ids'].dtype}")
print(f"   • attention_mask dtype: {batch['attention_mask'].dtype}")
print(f"   • labels dtype: {batch['labels'].dtype}")

print(f"\n4. EXEMPLES DU PREMIER BATCH:")
for i in range(min(3, 8)):
    input_len = batch['attention_mask'][i].sum().item()
    label_len = (batch['labels'][i] != -100).sum().item()
    print(f"   Exemple {i+1}:")
    print(f"     • Input: {input_len} tokens")
    print(f"     • Label: {label_len} tokens")

print(f"\n" + "=" * 70)
print("✅ TEST REUSSI: DataLoader fonctionne correctement !")
print("=" * 70)
