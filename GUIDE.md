# Guide de lecture

Ce guide suit un batch de données du fichier brut jusqu'au tableau de comparaison, en nommant à chaque étape le fichier qui fait le travail. Il ne remplace pas le [rapport](RAPPORT.md), qui présente les résultats : il explique le code qui les produit.

```text
corpus  ->  tokenisation  ->  Transformer  ->  loss  ->  backward  ->  optimiseur
                                                                          |
   comparaison  <-  MLflow  <-  métriques  <-  validation  <-  checkpoint <+
```

Le code est en anglais, ce guide en français. Chaque section renvoie au fichier concerné : le guide donne l'intention et la forme des tenseurs, le code donne le détail, et les deux ne doivent jamais se contredire. En cas de doute, le code a raison.

Pour voir les tenseurs réels traverser le modèle, exécutez [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) : il reprend les étapes de la section 3 en affichant les formes à chaque passage.

Les valeurs numériques citées sont celles de `scratch_100`, l'expérience de référence, déclarée dans [configs/experiments/scratch_100.yaml](configs/experiments/scratch_100.yaml).

---

## 1. Du dataset au corpus figé

**Répertoire :** [src/data/](src/data/) — **Commande :** `make data`

XSum compte environ 204 000 exemples. Le projet en fige 22 000 sous la graine 42 : 20 000 pour l'entraînement, 1 000 pour la validation, 1 000 pour le test. Le « 100 % » des ablations désigne ces 20 000 exemples, jamais XSum complet.

| Étape | Fichier | Ce qui se passe |
| --- | --- | --- |
| Téléchargement | [download.py](src/data/download.py) | Récupère XSum depuis Hugging Face |
| Nettoyage | [preprocess.py](src/data/preprocess.py) | Écarte les documents trop courts ou trop longs, normalise l'Unicode |
| Tirage | [split.py](src/data/split.py) | Tire les trois splits sous graine, calcule leurs empreintes |
| Écriture | [dataset.py](src/data/dataset.py) | Écrit trois `.jsonl` et un `manifest.json` |
| Contrôle | [validate.py](src/data/validate.py) | Vérifie qu'aucun document ne fuit d'un split à l'autre |

Un exemple est un triplet minimal, [example.py](src/data/example.py) :

```python
Example(example_id="35232142", source="Le document complet...", target="Le résumé de référence.")
```

Deux propriétés comptent pour la suite.

**Le corpus est figé, pas régénéré.** `manifest.json` porte un `dataset_version`, empreinte des trois splits. Chaque run enregistre cette empreinte. Deux scores calculés sous des empreintes différentes ne se comparent pas, et le dépôt le sait.

**Les sous-ensembles d'ablation sont emboîtés.** 10 % est un préfixe de 50 %, lui-même préfixe de 100 %. Tirés indépendamment, un écart entre deux points de la courbe mélangerait l'effet de la taille et celui de la composition de l'échantillon.

---

## 2. Tokenisation, `Dataset`, `DataLoader`, batchs

**Fichiers :** [src/data/tokenize.py](src/data/tokenize.py), [src/data/dataset.py](src/data/dataset.py), [src/training/sampler.py](src/training/sampler.py)

### Le tokenizer

Un modèle ne lit pas du texte, il lit des entiers. Le tokenizer T5 découpe le texte en sous-mots et rend un identifiant par sous-mot. **Les deux modèles du projet partagent ce tokenizer** : même vocabulaire, même découpage, même troncature. Un écart de score ne peut donc venir que des modèles.

```text
"summarize: Three men have been arrested."

21603   10    5245    1076   43      118     10195       5     1
summarize  :   Three   men   have   been   arrested     .   </s>
```

Neuf identifiants pour sept mots et deux ponctuations. Sur cet exemple chaque mot tient en un sous-mot, ce qui n'est pas garanti : un mot rare est découpé en plusieurs morceaux, et c'est exactement ce qui permet au vocabulaire de rester fini. Le `1` final est le token de fin de séquence. Le vocabulaire compte 32 100 entrées.

Le préfixe `"summarize: "` vient de la convention T5, qui multiplexe les tâches par un préfixe textuel. Le modèle from scratch le reçoit aussi, pour que l'entrée soit strictement identique.

### Ce que produit un batch

[`encode_batch`](src/data/tokenize.py) rend un `EncodedBatch` de quatre tenseurs :

| Champ | Forme | Rôle |
| --- | --- | --- |
| `input_ids` | `(8, S)` | Le document, tronqué à 512 tokens |
| `attention_mask` | `(8, S)` | 1 sur les vrais tokens, 0 sur le padding |
| `target_ids` | `(8, T)` | Le résumé, padding conservé |
| `labels` | `(8, T)` | Le résumé, padding remplacé par `-100` |

`S` et `T` sont les longueurs du plus long élément du batch, pas les plafonds : le padding est dynamique.

**Pourquoi `target_ids` et `labels` séparément.** `target_ids` sert à construire l'entrée du décodeur par décalage (section 3.9), `labels` sert à calculer la loss. La valeur `-100` est celle que `cross_entropy` ignore : sans elle, le modèle serait entraîné à prédire du padding, et la loss récompenserait le silence.

### Le sampler groupé par longueur

[`LengthGroupedSampler`](src/training/sampler.py) trie les exemples par longueur avant de former les batchs. Sur XSum ce n'est pas un détail : 38 % des documents atteignent le plafond de 512 tokens, donc un seul document long tire tout son batch à 512. La mesure est dans le fichier lui-même, obtenue par [scripts/measure_padding.py](scripts/measure_padding.py) :

```text
batch 8   mélangé : 24,8 % de padding      groupé : 1,5 %
```

Un quart du calcul de l'encodeur portait sur du vide. Le groupement conserve un aléa entre les mega-batchs, pour que l'ordre change à chaque époque.

---

## 3. Le Transformer, de bout en bout

**Répertoire :** [src/models/scratch/](src/models/scratch/) — un concept par fichier.

Configuration de référence : `d_model=256`, `num_heads=8`, donc `d_head=32`, 4 couches d'encodeur, 4 de décodeur, `d_ff=1024`. Vocabulaire T5 : 32 100 entrées. Total : 15,6 M paramètres, dont 53 % dans la seule table d'embedding.

Le tableau des formes, pour un batch de 8, un document de `S` tokens et un résumé de `T` tokens :

| Étape | Forme en sortie |
| --- | --- |
| `input_ids` | `(8, S)` |
| Embedding | `(8, S, 256)` |
| + encodage positionnel | `(8, S, 256)` |
| Projections Q, K, V | `(8, S, 256)` |
| Découpage en têtes | `(8, 8, S, 32)` |
| Scores d'attention | `(8, 8, S, S)` |
| Contexte | `(8, 8, S, 32)` |
| Recomposition des têtes | `(8, S, 256)` |
| Mémoire de l'encodeur | `(8, S, 256)` |
| États du décodeur | `(8, T, 256)` |
| Logits | `(8, T, 32100)` |

### 3.1 Embeddings

**Fichier :** [embeddings.py](src/models/scratch/embeddings.py)

Une table `(32100, 256)` associe un vecteur dense à chaque identifiant. C'est un paramètre : le modèle apprend ce que « chat » veut dire pour lui.

```python
Embedding(x) = E[x] * sqrt(d_model)
```

**La mise à l'échelle par `sqrt(d_model)` n'est pas cosmétique.** La table est initialisée avec un écart-type de `d_model ** -0.5`, soit environ 0,0625 ici. L'encodage positionnel qu'on va lui ajouter vit dans `[-1, 1]`. Sans le facteur `sqrt(256) = 16`, le signal de position écraserait le signal de token.

L'identifiant de padding a un embedding maintenu à zéro, qui ne reçoit aucun gradient.

### 3.2 Encodage positionnel

**Fichier :** [positional_encoding.py](src/models/scratch/positional_encoding.py)

L'attention est **invariante par permutation** : sans signal de position, « le chien mord l'homme » et « l'homme mord le chien » donneraient le même résultat. Le signal est ajouté, pas concaténé :

```text
PE(pos, 2i)     = sin(pos / 10000 ** (2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000 ** (2i / d_model))
```

Deux raisons à ce choix plutôt qu'une table apprise :

1. Il ne coûte aucun paramètre et se prolonge au-delà des longueurs vues à l'entraînement.
2. Pour un décalage `k` fixé, `PE(pos + k)` est une fonction **linéaire** de `PE(pos)`. Une position relative est donc atteignable par une application linéaire, ce qu'une tête d'attention calcule précisément.

La table est un buffer, pas un paramètre : elle suit le modèle sur le GPU et dans le checkpoint, mais l'optimiseur ne la touche jamais.

### 3.3 Query, Key, Value

**Fichier :** [multi_head_attention.py](src/models/scratch/multi_head_attention.py)

Trois projections linéaires du même vecteur d'entrée, trois rôles :

| Rôle | Question |
| --- | --- |
| **Query** | Ce que cette position cherche |
| **Key** | Ce que cette position propose |
| **Value** | Ce qu'on lit réellement quand cette position est retenue |

L'analogie utile est celle d'un dictionnaire : la requête est comparée à toutes les clés, et le résultat est une moyenne pondérée des valeurs. La différence avec un dictionnaire réel est que la sélection est **continue** : on ne prend pas une entrée, on prend un mélange de toutes.

Dans le code, une seule projection large par rôle, `nn.Linear(256, 256)`, suivie d'un redécoupage. C'est mathématiquement équivalent à huit petites projections et cela tient en un produit matriciel.

### 3.4 Scaled dot-product attention

**Fichier :** [attention.py](src/models/scratch/attention.py) — le cœur de l'architecture, 15 lignes de calcul.

```text
Attention(Q, K, V) = softmax(Q Kt / sqrt(d_k)) V
```

Trois opérations :

1. `Q Kt` donne à chaque requête un score de similarité contre chaque clé. Forme `(8, 8, S, S)` : c'est le seul tenseur dont la taille croît comme le **carré** de la longueur, et c'est pourquoi la troncature à 512 tokens coûte ce qu'elle coûte.
2. Le softmax, pris sur la dimension des clés, transforme chaque ligne de scores en distribution de probabilité.
3. Le produit avec `V` renvoie la moyenne pondérée des valeurs.

**Pourquoi diviser par `sqrt(d_k)`.** Si les composantes de `Q` et `K` sont indépendantes, centrées, de variance 1, leur produit scalaire sur `d_k` dimensions a une variance de `d_k`. Quand `d_k` grandit, les scores s'étalent, le softmax sature, et son gradient s'annule. Diviser par `sqrt(d_k)` ramène la variance à 1 et garde le gradient utilisable. Ici `d_k = 32`, donc on divise par environ 5,66.

**Self-attention ou cross-attention** n'est pas une autre fonction : c'est la même, appelée différemment. En self-attention, `Q`, `K` et `V` viennent du même tenseur. En cross-attention, `Q` vient du décodeur et `K`, `V` de l'encodeur.

### 3.5 Multi-head attention

**Fichier :** [multi_head_attention.py](src/models/scratch/multi_head_attention.py)

Une tête unique produit **une** moyenne pondérée, donc exprime **une** relation. Huit têtes en parallèle, chacune sur 32 dimensions :

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_8) W_O
```

Le coût total est celui d'une seule tête de largeur 256 : découper la largeur achète de l'expressivité gratuitement. Une tête peut suivre le sujet de la phrase pendant qu'une autre suit la position.

La concaténation est une opération de forme : `(8, 8, S, 32)` redevient `(8, S, 256)`. `W_O` mélange ensuite ce que les têtes ont récolté séparément, sans quoi elles resteraient dans des sous-espaces étanches.

### 3.6 Les masques

**Fichier :** [masks.py](src/models/scratch/masks.py) — convention : `True` autorise, `False` interdit.

**Masque de padding.** Les séquences d'un batch ont des longueurs différentes. Le padding ne porte aucune information : si le modèle pouvait le lire, ses prédictions dépendraient de la composition du batch.

**Masque causal.** Le décodeur est entraîné en *teacher forcing* : toute la cible lui est donnée d'un coup. La position `t` ne doit voir que les positions `<= t`, sinon le modèle lit la réponse qu'on lui demande de prédire. La loss s'effondre, et la génération reste aléatoire. C'est le bug le plus classique du domaine, et le plus silencieux.

Les deux se combinent pour le décodeur, [`build_decoder_mask`](src/models/scratch/masks.py) : ne pas regarder devant **et** ne pas regarder le padding. En cross-attention le masque est celui du **source** : chaque position du décodeur peut lire tout le document, seul le futur de la cible est interdit.

**Un détail qui vaut un test.** Les positions interdites reçoivent la plus petite valeur finie du type, pas moins l'infini. Sur une séquence entièrement composée de padding, moins l'infini donnerait une ligne de zéros divisée par zéro, donc des NaN.

### 3.7 Connexions résiduelles et normalisation

**Fichier :** [encoder_layer.py](src/models/scratch/encoder_layer.py)

`x + Sublayer(x)` donne au gradient un chemin qui contourne entièrement le sous-bloc. Sans les résiduels, une pile de six couches s'entraîne mal et une pile plus profonde pas du tout.

La normalisation de couche ramène chaque position à une moyenne nulle et une variance unité **sur ses features**, ce qui stabilise l'échelle du flux résiduel d'une couche à l'autre. Deux placements existent :

```text
post-norm   x = LayerNorm(x + Sublayer(x))      article original
pre-norm    x = x + Sublayer(LayerNorm(x))      défaut du projet
```

Le pre-norm laisse le chemin résiduel libre de toute normalisation, donc le gradient atteint la première couche sans distorsion. Il s'entraîne de façon fiable sans le long warmup que le post-norm exige, ce qui compte pour un modèle parti de zéro sur un petit corpus. `norm_first=False` restitue la disposition de l'article.

### 3.8 Feed-forward

**Fichier :** [feed_forward.py](src/models/scratch/feed_forward.py)

Deux couches linéaires avec une activation entre les deux, `256 -> 1024 -> 256`, appliquées **indépendamment à chaque position**. C'est ce que veut dire « position wise » : aucune information ne circule entre les positions ici, ce travail appartient entièrement à l'attention.

L'attention mélange, le feed-forward transforme. Les deux alternent.

### 3.9 Encodeur et décodeur

**Fichiers :** [encoder.py](src/models/scratch/encoder.py), [decoder.py](src/models/scratch/decoder.py), [decoder_layer.py](src/models/scratch/decoder_layer.py), [transformer.py](src/models/scratch/transformer.py)

Une couche d'encodeur enchaîne deux sous-blocs : self-attention, puis feed-forward. Une couche de décodeur en enchaîne **trois** :

1. **Self-attention masquée** sur ce qui a déjà été écrit.
2. **Cross-attention** : les requêtes viennent du décodeur, les clés et valeurs de la mémoire de l'encodeur. C'est le point de rencontre des deux tours. Le décodeur demande : compte tenu de ce que j'ai écrit, quelle partie du document dois-je lire maintenant ?
3. **Feed-forward**, identique à celui de l'encodeur.

L'encodeur lit le document **une fois** et produit une mémoire `(8, S, 256)` que le décodeur consulte à chacune de ses positions.

**Le décalage à droite.** [`shift_target_right`](src/models/scratch/transformer.py) construit l'entrée du décodeur en décalant la cible d'une position et en préfixant le token de départ :

```text
cible           [ Three, men, have, been, arrested ]
entrée décodeur [ <start>, Three, men, have, been  ]
```

La position `t` du décodeur contient donc le token `t - 1` de la cible : prédire le token `t` ne demande que ce que le masque causal autorise à voir. Le tokenizer T5 n'a pas de token de début de séquence, donc l'identifiant de padding joue ce rôle, exactement comme T5 lui-même.

### 3.10 Projection finale vers le vocabulaire

**Fichier :** [decoder.py](src/models/scratch/decoder.py)

Le dernier état caché de chaque position, de dimension 256, est projeté sur 32 100 scores, un par entrée du vocabulaire. Ces scores sont les **logits** : ce ne sont pas encore des probabilités, le softmax de la loss ou de la génération s'en charge.

**Poids liés.** Quand `tie_embeddings` est vrai, la projection réutilise la transposée de la table d'embedding. Cela économise `vocab_size * d_model` paramètres, soit 8,2 M ici, et régularise un modèle entraîné sur peu de données en forçant la vue d'entrée et la vue de sortie d'un token à s'accorder.

### 3.11 Génération

**Fichier :** [generation.py](src/models/scratch/generation.py)

À l'entraînement, tout le résumé est fourni d'un coup. À la génération, rien n'est fourni : le modèle produit un token, se le redonne, et recommence. C'est l'autorégression.

| Stratégie | Principe |
| --- | --- |
| `greedy_search` | Prend le token le plus probable à chaque pas |
| `beam_search` | Garde les `k` meilleures séquences partielles, `k = 4` ici |
| `sample_search` | Échantillonne, avec température et top-k/top-p |

L'évaluation du projet utilise le **beam search à 4 faisceaux**, jamais l'échantillonnage : un score mesuré sur des sorties aléatoires ne serait pas reproductible. `no_repeat_ngram_size=3` interdit la répétition d'un trigramme, défaut classique des modèles sous-entraînés.

---

## 4. Une étape d'entraînement

**Fichier principal :** [src/training/trainer.py](src/training/trainer.py) — **Commande :** `make train-scratch`

Ce qui suit décrit ce qui arrive à **un** batch, dans l'ordre.

### 4.1 Forward pass

Le batch part sur le device, puis traverse le modèle. La fonction de loss est un paramètre du trainer, [`BatchLossFn`](src/training/trainer.py) : c'est ce qui permet au même trainer de piloter le Transformer from scratch et T5, dont les signatures diffèrent.

```python
output = model(batch.input_ids, target_ids=batch.target_ids)   # logits (8, T, 32100)
```

### 4.2 La loss

Entropie croisée entre les logits et les labels, [`make_scratch_batch_loss`](src/training/trainer.py) :

```python
F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),   # (8 * T, 32100)
    batch.labels.reshape(-1),              # (8 * T,)
    ignore_index=IGNORE_INDEX,             # -100 : le padding ne compte pas
    label_smoothing=label_smoothing,       # 0,1
)
```

Les deux tenseurs sont aplatis en une prédiction par ligne. Chaque position est notée indépendamment, et les positions de padding sont retirées de la moyenne.

**Le lissage de labels** retire 10 % de la masse au token de référence et l'étale sur le vocabulaire. Le modèle est ainsi pénalisé s'il devient trop confiant, ce qui réduit le surapprentissage sur un petit corpus.

**La loss est pondérée par les tokens, pas par les batchs.** Un batch contient un nombre variable de tokens cibles : une moyenne simple sur les batchs pondérerait un résumé court comme un résumé long, et la courbe d'entraînement ne serait plus comparable à celle de validation.

### 4.3 Rétropropagation

```python
torch.autograd.backward(self.scaler.scale(loss / accumulation))
```

`backward()` calcule le gradient de la loss par rapport à chaque paramètre et l'**accumule** dans `parameter.grad`. Rien n'est encore modifié : le gradient dit dans quelle direction chaque poids devrait bouger.

La division par `accumulation` vient de ce que le gradient d'une moyenne est la moyenne des gradients. Avec `batch_size=8` et `gradient_accumulation_steps=4`, le modèle voit un batch effectif de 32 exemples tout en n'en tenant que 8 en mémoire à la fois.

### 4.4 L'optimiseur et la mise à jour des poids

**Fichier :** [src/training/optimizer.py](src/training/optimizer.py)

Toutes les 4 accumulations, [`_optimiser_step`](src/training/trainer.py) exécute la séquence complète :

```text
1. unscale_        annule la mise à l'échelle de la précision mixte
2. clip_grad_norm_ borne la norme globale du gradient à 1,0
3. scaler.step     AdamW met à jour les poids
4. scheduler.step  avance le learning rate
5. zero_grad       remet les gradients à zéro
```

**AdamW plutôt qu'Adam.** Les deux n'appliquent pas la décroissance de poids de la même façon. Adam la replie dans le gradient, donc le dénominateur adaptatif la remet à l'échelle et les paramètres à faible gradient finissent à peine régularisés. AdamW l'applique directement aux poids, ce qui est le comportement attendu de l'hyperparamètre.

**La décroissance ne porte que sur les matrices.** Biais et gains de normalisation sont unidimensionnels et agissent comme des décalages : les tirer vers zéro contraint la représentation sans bénéfice de régularisation.

**Le clipping est mesuré avant d'être appliqué**, et la norme est rapportée. Un gradient qui explose est ainsi visible dans les logs plutôt que silencieusement écrêté.

### 4.5 Learning rate et scheduler

**Fichier :** [src/training/scheduler.py](src/training/scheduler.py)

Le pas d'apprentissage n'est pas constant. Toutes les formes commencent par un **warmup linéaire** : un Transformer parti de zéro est fragile dans ses premiers pas, les logits d'attention sont quasi uniformes et les gradients grands, donc partir au taux maximal diverge régulièrement.

```text
linear        warmup, puis droite jusqu'à zéro         <- utilisé ici
cosine        warmup, puis demi-cosinus
inverse_sqrt  warmup, puis sqrt(warmup / step)         <- schéma de l'article
constant      warmup, puis palier
```

Ici : taux maximal `3e-4`, `warmup_ratio=0.06`, soit 6 % des pas totaux consacrés à la montée.

**Le scheduler avance par pas d'optimisation, jamais par époque.** Une avance par époque rendrait le schéma dépendant de la taille du corpus, et l'ablation sur la taille mesurerait alors deux choses à la fois.

### 4.6 Époques et validation

Une époque est un passage complet sur le split d'entraînement. À la fin de chaque époque, [`Trainer.train`](src/training/trainer.py) :

1. calcule la loss de validation avec la même mesure pondérée par tokens ;
2. demande à [`EarlyStopping`](src/training/early_stopping.py) si elle s'est améliorée ;
3. écrit un checkpoint, marqué « best » si oui ;
4. s'arrête si la patience est épuisée, ici 3 époques sans amélioration.

**Pourquoi l'arrêt anticipé compte ici.** Un Transformer from scratch sur 20 000 exemples surapprend bien avant la dixième époque. Aller au bout rapporterait le score d'un modèle surappris, et l'ablation sur la taille du corpus mesurerait la patience plutôt que les données.

La validation ne sert **qu'**à cette décision et au suivi. Le score publié est mesuré sur le split de test, que rien n'a consulté pendant l'entraînement.

### 4.7 Checkpoints

**Fichier :** [src/training/checkpoint.py](src/training/checkpoint.py)

Un checkpoint ne contient pas que les poids. Il contient tout ce qu'il faut pour reprendre : moments de l'optimiseur, position du scheduler, état du gradient scaler, compteurs de l'arrêt anticipé, et l'état de chaque générateur aléatoire. Restaurer les seuls poids produirait un modèle différent d'un entraînement ininterrompu.

Deux garde-fous valent d'être signalés :

- Le contenu est fait de tenseurs et de primitives uniquement, donc il se recharge avec `weights_only=True`. Un checkpoint est une donnée : le charger ne doit jamais pouvoir exécuter du code.
- L'écriture passe par un fichier temporaire puis un renommage. Une coupure pendant l'écriture laisse le checkpoint précédent intact plutôt qu'un fichier tronqué.

### 4.8 CPU et GPU

**Fichier :** [src/utils/device.py](src/utils/device.py)

`device: auto` prend le GPU s'il y en a un. La **précision mixte** n'est activée que sur CUDA : les activations passent en float16, les poids restent en float32, et un `GradScaler` multiplie la loss pour éviter que les petits gradients ne s'annulent en float16. Sur CPU elle n'apporte rien et ajoute des conversions.

Un poste sans GPU exécute tout, en plus lent. Les durées cessent en revanche d'être comparables à celles de `reports/`.

### 4.9 Reproductibilité

**Fichier :** [src/utils/seed.py](src/utils/seed.py)

[`set_seed`](src/utils/seed.py) est appelée une fois par expérience, avant la construction des chargeurs et du modèle. Elle fixe `random`, `numpy`, `torch`, `PYTHONHASHSEED`, et met cuDNN en mode déterministe.

C'est ce dont dépend toute la comparaison du projet : sans elle, deux runs de la même configuration donneraient deux scores, et l'ablation ne mesurerait plus la taille du corpus.

> **Limite honnête.** Chaque configuration n'a été jouée que sous la graine 42. Les intervalles publiés sont des intervalles bootstrap sur le jeu de test : ils mesurent l'échantillonnage de l'évaluation, pas la variance d'entraînement. La section 7 du [rapport](RAPPORT.md) dit lesquelles des conclusions en dépendent.

---

## 5. Les métriques

**Fichiers :** [src/metrics/rouge.py](src/metrics/rouge.py), [src/evaluation/evaluator.py](src/evaluation/evaluator.py)

ROUGE mesure le recouvrement entre le résumé produit et le résumé de référence.

| Variante | Ce qu'elle compte |
| --- | --- |
| ROUGE-1 | Unigrammes communs |
| ROUGE-2 | Bigrammes communs |
| ROUGE-L | Plus longue sous-séquence commune |

ROUGE-L est la métrique rapportée : elle tolère les réordonnancements, ce qu'un résumé abstractif fait constamment.

**Chaque score porte un intervalle de confiance.** Le calcul rééchantillonne 1 000 fois les documents notés et rend l'intervalle de percentiles à 95 %, sous la forme d'un [`ConfidenceInterval`](src/metrics/rouge.py). Un score seul ne dit pas s'il diffère de son voisin, et deux intervalles qui se recouvrent ne permettent pas de conclure. Le rapport applique cette règle, y compris quand elle l'empêche de conclure.

L'évaluation est faite sur les mêmes 1 000 documents de test pour toutes les expériences, sans quoi les scores ne se compareraient pas.

---

## 6. MLflow

**Répertoire :** [src/tracking/](src/tracking/) — **Interface :** `make mlflow-ui`, sur <http://localhost:5000>

MLflow répond à une question : quel run a produit ce score, avec quelle configuration, sur quel corpus, depuis quel commit.

### Ce qui est enregistré, quand

| Moment | Ce qui part vers MLflow | Fichier |
| --- | --- | --- |
| Avant l'entraînement | Ouverture du run, sous le nom de l'expérience | [live.py](src/tracking/live.py) |
| Pendant | `step_loss`, `step_learning_rate`, `step_grad_norm`, puis `epoch_train_loss` et `epoch_validation_loss` | [live.py](src/tracking/live.py) |
| Après | Paramètres, métriques finales, tags, artefacts | [payload.py](src/tracking/payload.py) |

**Paramètres** : ce que l'expérience a déclaré et qu'un rejeu devrait répéter — graine, proportion de corpus, architecture, hyperparamètres d'entraînement et de décodage.

**Métriques** : ROUGE-1, ROUGE-2, ROUGE-L, durée d'entraînement, meilleure loss de validation.

**Tags** : où le run a tourné — GPU, version de torch, commit git, propreté de l'arbre de travail. Un run qui rapporte les mêmes paramètres sur une autre machine est la même expérience, d'où la séparation.

**Artefacts** : `run.json`, `metrics.json`, `history.json`, `qualitative.json`. Quatre petits fichiers texte. Les prédictions et les checkpoints ne sont **pas** copiés : les poids vivent déjà sous `runs/`, et les recopier créerait une seconde copie à l'endroit où rien ne les charge.

### Trois règles qui expliquent le code

**Un échec de tracking ne fait jamais échouer un run.** Le résultat d'une expérience est l'enregistrement sur disque ; le magasin en est un miroir. Un serveur injoignable ne doit pas transformer six heures d'entraînement en plantage après que la mesure a été prise. Tout passe par `log_safely`.

**Rien n'est enregistré qui n'ait été écrit d'abord.** Le lanceur écrit `run.json`, puis le trace. Le magasin ne contient donc jamais un run dont le dépôt n'a pas trace. Les séries live sont l'exception assumée, et elles portent d'autres noms : `step_*` et `epoch_*` sont ce que la boucle a observé, les métriques finales viennent de l'enregistrement seul. Les mélanger reviendrait à lire une observation comme un résultat.

**Le tracking se désactive, jamais ne se simule.** `--no-tracking` sélectionne [`NullTracker`](src/tracking/client.py), qui renvoie `None` et le dit. Aucun mode ne prétend avoir enregistré.

### Rejouer une expérience depuis MLflow

Un run porte son `git_commit`, son `dataset_version` et sa configuration complète. Reproduire consiste à revenir au commit, vérifier que l'empreinte du corpus correspond, et relancer :

```bash
python -m src.experiments.run --config configs/experiments/scratch_100.yaml
```

`python -m src.tracking.log --all` renvoie vers MLflow des enregistrements déjà écrits, par exemple après un run lancé hors ligne.

---

## 7. Comparer les expérimentations

**Répertoire :** [src/experiments/](src/experiments/) — le seul module que ce guide ne détaille pas.

Il porte la machinerie de campagne : lecture des fichiers d'expérience, boucle sur les neuf runs, ablations, tableaux, figures. Ce n'est pas de l'apprentissage automatique, c'est de l'orchestration, et on peut comprendre tout le reste sans l'ouvrir.

Ce qu'il faut en savoir tient en quatre commandes :

| Commande | Effet |
| --- | --- |
| `make reproduce MODE=quick` | Joue toute la chaîne en une minute, plafonnée. Ne produit aucun résultat, et le dit |
| `make reproduce MODE=full` | Rejoue la campagne réelle, plusieurs heures |
| `make ablation` | Rejoue les deux ablations et écrit leurs tableaux |
| `make report-sync` | Régénère les tableaux du rapport depuis les enregistrements |

Un enregistrement porte un statut, et le statut est ce qui protège les conclusions :

| Statut | Sens |
| --- | --- |
| `OK` | Mesure complète, entre dans les tableaux |
| `PARTIAL` | Run plafonné ou évalué sur un sous-ensemble. N'entre dans aucun tableau |
| `FAILED` | L'expérience a levé. L'erreur est enregistrée |
| `NOT_RUN` | Déclarée, jamais exécutée |

Une valeur factice ne devient jamais un résultat : c'est la règle que ces statuts font respecter mécaniquement.

---

## 8. Par où commencer

Dans cet ordre, en une soirée :

1. `notebooks/00_environment_check.ipynb` — ce poste peut-il exécuter la chaîne, et sur quoi.
2. `make data` — construire le corpus, une fois.
3. [notebooks/03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) — voir un batch réel traverser le modèle, forme par forme.
4. `notebooks/02_training.ipynb` en `MODE = "quick"` — lancer une expérience en quelques secondes et voir ses courbes.
5. `make mlflow-ui` — retrouver le run dans le magasin.
6. Le [rapport](RAPPORT.md) — ce que les neuf expériences ont montré.

Les tests sont l'autre porte d'entrée : [tests/unit/test_scratch_attention.py](tests/unit/test_scratch_attention.py) et [tests/unit/test_scratch_masks.py](tests/unit/test_scratch_masks.py) vérifient les propriétés énoncées en section 3, et échoueraient si elles étaient fausses.
