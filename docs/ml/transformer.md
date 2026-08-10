# Transformer from scratch

Le modèle est un Transformer encodeur-décodeur écrit composant par composant en PyTorch. `torch.nn.Transformer` n'est pas utilisé : l'objectif est de démontrer la compréhension de l'architecture, pas d'en consommer une implémentation.

!!! tip "Première lecture"
    [Transformer expliqué simplement](transformer-eli5.md) parcourt la même architecture sur un exemple unique, sans formalisme. Cette page-ci en donne la version exacte.

## Composants

```text
src/models/scratch/
├── config.py                 hyperparamètres validés
├── embeddings.py             table d'embedding, mise à l'échelle par sqrt(d_model)
├── positional_encoding.py    encodage sinusoïdal fixe
├── attention.py              softmax(Q Kt / sqrt(d_k)) V
├── multi_head_attention.py   projections, découpage en têtes, concaténation
├── feed_forward.py           réseau position par position
├── masks.py                  masque de padding, masque causal
├── encoder_layer.py          self-attention + feed forward, résiduels
├── decoder_layer.py          self-attention masquée + cross-attention + feed forward
├── encoder.py                pile d'encodeurs
├── decoder.py                pile de décodeurs + projection vers le vocabulaire
├── transformer.py            assemblage, teacher forcing, perte
└── generation.py             génération autorégressive, greedy et beam search
```

## La formule d'attention

```text
Attention(Q, K, V) = softmax(Q Kt / sqrt(d_k)) V
```

| Terme | Rôle |
| --- | --- |
| **Q**, requêtes | Un vecteur par position de sortie : ce que cette position cherche |
| **K**, clés | Un vecteur par position d'entrée : ce que cette position propose |
| **V**, valeurs | Un vecteur par position d'entrée : ce qui est effectivement lu |
| **d_k** | Dimension d'une tête, soit `d_model / num_heads` |

`Q Kt` attribue à chaque requête un score de similarité contre chaque clé. Le softmax transforme chaque ligne de scores en distribution de probabilité, et le produit avec `V` renvoie une moyenne pondérée des valeurs.

### Pourquoi diviser par la racine de d_k

Si les composantes de `Q` et `K` sont indépendantes, centrées et de variance unité, leur produit scalaire sur `d_k` dimensions a une variance de `d_k`. Quand `d_k` grandit, les scores s'étalent, le softmax sature et son gradient s'annule. La division ramène la variance à un et garde le gradient exploitable.

Le test `test_scores_are_divided_by_the_square_root_of_d_k` vérifie cette division sur une entrée construite à la main.

### Concaténation des têtes

Une tête unique moyenne les valeurs qu'elle sélectionne : elle n'exprime qu'une relation à la fois. Plusieurs têtes travaillent en parallèle sur des sous-espaces de dimension `d_model / num_heads`, si bien que le coût total égale celui d'une tête unique de largeur `d_model`.

L'implémentation applique une projection large par rôle, puis la redécoupe en têtes. C'est mathématiquement équivalent à `h` petites projections, et cela tient en un seul produit matriciel. La concaténation est une simple opération de forme : la dimension des têtes revient contre celle des traits, et `W_O` mélange ensuite ce que les têtes ont collecté séparément.

## Les masques

**Masque de padding.** Les séquences d'un lot ont des longueurs différentes, complétées jusqu'à la plus longue. Sans masque, le modèle apprendrait à lire le remplissage et ses prédictions dépendraient de la composition du lot.

**Masque causal.** L'entraînement se fait par teacher forcing : la cible entière est fournie d'un coup. La position `t` ne doit voir que les positions `<= t`, sinon le modèle lit la réponse qu'on lui demande de prédire, la perte s'effondre et la génération reste aléatoire.

Les deux masques suivent la convention `True` vaut conserver, `False` vaut interdire, et se diffusent vers la forme `(batch, têtes, requêtes, clés)`.

Les positions interdites reçoivent la plus petite valeur finie du type, pas moins l'infini. Sur une séquence entièrement remplie de padding, moins l'infini produirait une ligne de zéros divisée par zéro, donc des NaN.

## Résiduels et normalisation

Deux placements existent :

```text
post-norm  x = LayerNorm(x + Sublayer(x))     article original
pre-norm   x = x + Sublayer(LayerNorm(x))     défaut du projet
```

Le pre-norm laisse le chemin résiduel libre de toute normalisation, si bien que le gradient atteint la première couche sans distorsion. Il s'entraîne de façon fiable sans le long warmup que le post-norm réclame, ce qui compte pour un modèle entraîné from scratch sur un petit corpus.

Le post-norm reste accessible par `norm_first: false`, pour une reproduction fidèle de l'article. Les deux dispositions sont couvertes par les tests.

## Partage des poids

Une seule table d'embedding sert l'encodeur, le décodeur et, par défaut, la projection de sortie. Le partage source-cible se justifie ici parce que les deux vivent dans le même vocabulaire, celui de T5.

L'attachement de la projection de sortie économise `vocab_size * d_model` paramètres, soit environ 16 millions avec le vocabulaire T5 et `d_model = 512`. Il régularise aussi un modèle entraîné sur un petit corpus, en forçant les vues d'entrée et de sortie d'un token à s'accorder.

## Teacher forcing

Le décodeur reçoit la cible décalée d'une position vers la droite, préfixée par `decoder_start_token_id`, et prédit la cible non décalée.

```text
cible           [ le, chat, dort, </s> ]
entrée décodeur [ <s>, le, chat, dort ]
```

Le tokenizer T5 n'a pas de token de début de séquence. Comme T5 lui-même, le projet utilise l'identifiant de padding pour démarrer le décodeur.

## Génération

**Greedy search** prend le token le plus probable à chaque pas. C'est la référence honnête : tout gain annoncé avec beam search doit être mesuré contre elle.

**Beam search** conserve les `k` séquences partielles les plus probables. Greedy est myope : un token qui paraît optimal maintenant peut mener à une mauvaise séquence. Le score d'un faisceau terminé est la somme des log-probabilités divisée par `longueur ** length_penalty`, sans quoi les séquences courtes seraient mécaniquement favorisées.

Deux contraintes sont disponibles :

```text
min_new_tokens        interdit la fin de séquence avant ce nombre de tokens
no_repeat_ngram_size  interdit de répéter un n-gramme déjà produit
```

La seconde traite un défaut classique des décodeurs mal entraînés : la boucle sur un même fragment.

Ces hyperparamètres vivent dans `src/models/generation.py`, que la baseline pré-entraînée lit également. Les deux modèles décodent donc sous les mêmes contraintes, et un écart de score ne peut venir que des modèles. Voir [Baseline pré-entraînée](baseline.md).

!!! note "Pas de cache clé valeur"
    Chaque pas redécode le préfixe entier. Avec un budget de 64 tokens le coût reste modeste et le code reste lisible, ce qui importe davantage ici que le débit. Ajouter un cache changerait la vitesse, jamais la sortie.

## Configuration par défaut

```yaml
d_model: 512
num_heads: 8
num_encoder_layers: 6
num_decoder_layers: 6
d_ff: 2048
dropout: 0.1
max_position: 1024
tie_embeddings: true
norm_first: true
```

`d_model` doit être divisible par `num_heads`, et pair pour que l'encodage positionnel puisse remplir ses dimensions par paires sinus et cosinus. Les deux contraintes sont vérifiées à la construction.

## Ce que les tests garantissent

Deux propriétés ne se voient pas sur une courbe de perte et ne sont attrapées que par un test explicite.

**Le décodeur ne lit pas le futur.** Modifier le token de l'entrée décodeur à la position `k` laisse les logits des positions `< k` strictement inchangés, et modifie ceux de la position `k`. La seconde moitié de l'assertion compte autant que la première : sans elle, un modèle qui ignorerait entièrement son entrée passerait le test.

**Le padding ne change rien.** Ajouter du remplissage à la source laisse les logits identiques. Un masque mal diffusé ferait échouer ce test alors que la perte continuerait de descendre.

S'y ajoutent :

```text
les poids d'attention forment une distribution de probabilité
une position masquée reçoit un poids nul, et la valeur qu'elle porte devient sans effet
une ligne entièrement masquée ne produit pas de NaN
l'encodage positionnel reproduit la formule publiée, position par position
le feed forward traite chaque position indépendamment
l'attachement des embeddings économie exactement vocab_size * d_model paramètres
chaque paramètre reçoit un gradient
le modèle surapprend un lot unique, la perte tombe sous 20 % de sa valeur initiale
```

Le dernier est le contrôle de bout en bout le plus fort disponible sans entraînement réel : si la perte ne s'effondre pas sur un lot unique, le câblage est défectueux quelque part.

## Référence de code

::: src.models.scratch.attention

::: src.models.scratch.masks
