# Documentation des Problèmes et Solutions - Pretraitement des Donnees

## 📋 Vue d'ensemble

Ce document detaille les problemes rencontrres lors du pretraitement du dataset CNN/DailyMail et les solutions implrmentees.

**Dataset** : CNN/DailyMail (version 3.0.0)  
**Taille** : ~311,971 exemples (~835 MB)  
**Objectif** : Preparer les données pour l'entraAnement du Transformer from scratch et T5

---

## ❌ Problème 1 : Erreur de Mmoire (ArrowMemoryError)

### Description

Lors de la tokenization du split `train_50` (143,556 exemples), l'erreur suivante est survenue :

```
ArrowMemoryError: realloc of size 1073741824 failed
```

### Cause Racine

La fonction originale tentait de :
1. Concatrner tous les batches tokenisrs dans un DataFrame pandas gérant
2. Sauvegarder en un seul fichier Parquet

**Problème** : Le DataFrame final nércessitait 1 GB de mmoire continue, non disponible.

### Solution Implrmentee

**Remplacer Pandas + Parquet par HuggingFace Datasets + Sharding**

```python
# AVANT (rchec)
df_final = pd.concat(tokenized_list)
df_final.to_parquet(output_path)

# APRRS (succrs)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.save_to_disk(output_path, num_shards=14)
```

### Rrsultats

| Split | Avant | Aprrs |
|-------|-------|-------|
| train_10 | ✅ | ✅ 2 shards |
| train_50 | ❌ Erreur | ✅ 14 shards |
| train_100 | Non testr | ✅ 28 shards |
| validation | ✅ | ✅ 1 shard |
| test | ✅ | ✅ 1 shard |

**Total** : 484,238 exemples, 1.42 GB, 46 shards

---

## ✅ Avantages de la Solution

1. **Pas d'erreur mmoire** : Format Arrow optimisr
2. **Sauvegarde rapide** : < 3 secondes par split
3. **Chargement rapide** : `load_from_disk()` optimisr
4. **Scalable** : Fonctionne avec des datasets plus volumineux
5. **Standard** : Recommandr par HuggingFace

---

## 🔄 Workflow Final

1. **Nettoyage** : `clean_text()` sur articles et rrsumrs
2. **Tokenization** : SentencePiece pour Transformer, T5 tokenizer pour T5
3. **Sauvegarde** : `dataset.save_to_disk()` avec sharding
4. **Vrification** : `load_from_disk()` et contrôles

---

## 📝 Notes

- **Sharding** : 1 shard par 10,000 exemples
- **Format** : HuggingFace Datasets (Arrow)
- **Mmoire** : Libération systrmatique avec `gc.collect()`
