# Text Summarization Comparison

**Automatic Text Summarization: Transformer From Scratch vs Pre-trained T5**

## 📋 Description

Ce projet de Master en NLP compare deux approches de résumé automatique de texte :

1. **Transformer From Scratch** : Implémentation complète d'un Transformer encodeur-décodeur entraîné from scratch
2. **T5 Pré-entraîné** : Fine-tuning du modèle T5 pré-entraîné de Hugging Face

L'objectif est d'étudier l'impact de la taille du corpus d'entraînement sur les performances des deux approches, en utilisant le dataset CNN/DailyMail.

## 🎯 Objectifs

- Implémenter un Transformer encodeur-décodeur from scratch avec attention multi-tête
- Fine-tuner le modèle T5 pré-entraîné sur le dataset CNN/DailyMail
- Comparer les performances (ROUGE, BLEU) des deux approches
- Analyser l'impact de la taille du corpus d'entraînement
- Évaluer la qualité des résumés générés

## 🗂️ Structure du Projet

```
text-summarization-comparison/
├── configs/                 # Fichiers de configuration YAML
│   ├── default.yaml        # Configuration par défaut
│   ├── training.yaml       # Paramètres d'entraînement
│   ├── transformer.yaml    # Config du Transformer from scratch
│   └── t5.yaml            # Config du modèle T5
├── data/                   # Données du projet
│   ├── raw/               # Données brutes (CNN/DailyMail)
│   ├── processed/         # Données prétraitées
│   ├── tokenized/         # Données tokenizées
│   └── cache/             # Cache des données
├── notebooks/             # Jupyter notebooks pour l'exploration
│   ├── 01_dataset_download.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── playground.ipynb
├── src/                   # Code source
│   ├── data/             # Modules de gestion des données
│   │   ├── downloader.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── models/           # Modèles
│   │   ├── transformer/  # Transformer from scratch
│   │   └── t5/          # T5 pré-entraîné
│   ├── evaluation/       # Métriques d'évaluation
│   ├── experiments/      # Scripts d'entraînement
│   ├── api/             # API REST (optionnel)
│   ├── app/             # Application web (optionnel)
│   └── utils/           # Fonctions utilitaires
├── tests/                # Tests unitaires
├── outputs/              # Résultats
│   ├── checkpoints/     # Modèles entraînés
│   ├── figures/         # Graphiques et visualisations
│   ├── predictions/     # Prédictions du modèle
│   └── reports/         # Rapports d'évaluation
├── requirements.txt      # Dépendances Python
├── pyproject.toml       # Configuration du projet
└── docker-compose.yml   # Configuration Docker
```

## 🚀 Installation

### Prérequis

- Python 3.11+
- pip ou conda
- (Optionnel) Docker et docker-compose

### Installation locale

```bash
# Cloner le repository
git clone https://github.com/skai-shiroe/Train-vs-Pre-train.git
cd Train-vs-Pre-train/text-summarization-comparison

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

### Installation avec Docker

```bash
# Construire et lancer les services
docker-compose up --build
```

## 📊 Dataset

**CNN/DailyMail** (version 3.0.0)
- ~300k articles de news avec résumés
- Split : train / validation / test
- Tâche : Résumé extractif et abstractif

## 🔬 Méthodologie

### 1. Transformer From Scratch
- Architecture encodeur-décodeur complète
- Attention multi-tête
- Encodage positionnel sinusoidal
- Entraînement from scratch sur CNN/DailyMail

### 2. T5 Pré-entraîné
- Fine-tuning de T5-small/base/large
- Utilisation du tokenizer T5
- Transfer learning depuis Hugging Face

### 3. Évaluation
- Métriques : ROUGE-1, ROUGE-2, ROUGE-L, BLEU
- Comparaison des performances
- Analyse de l'impact de la taille du corpus

## 🛠️ Technologies

- **Deep Learning** : PyTorch 2.0+, Transformers
- **NLP** : Hugging Face Datasets, Tokenizers
- **Évaluation** : ROUGE, BLEU
- **Visualisation** : Matplotlib, Seaborn
- **Configuration** : PyYAML

## 📈 Résultats Attendus

- Comparaison quantitative des performances
- Analyse de l'impact de la taille du corpus
- Visualisation des courbes d'apprentissage
- Exemples de résumés générés

## 📝 License

MIT License - Voir le fichier LICENSE pour plus de détails.

## 👤 Auteur

Projet de Master en NLP - 2024

## 📚 Références

- Vaswani et al. (2017) - "Attention Is All You Need"
- Raffel et al. (2020) - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- CNN/DailyMail Dataset: https://huggingface.co/datasets/cnn_dailymail