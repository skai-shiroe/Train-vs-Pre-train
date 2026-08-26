# Rapport

Entraîner un Transformer encodeur-décodeur pour le résumé automatique, puis le comparer à un modèle pré-entraîné de référence sur les mêmes données de test.

| Exigence | Ce qui a été fait |
| --- | --- |
| Modèle from scratch | Transformer encodeur-décodeur écrit bloc par bloc, 60 575 744 paramètres, entraîné sur 10 %, 50 % et 100 % du corpus |
| Témoin d'initialisation | Architecture `t5-small` épinglée, 60 506 624 paramètres tirés au hasard, sur les mêmes trois proportions |
| Modèle pré-entraîné, zero-shot puis fine-tuné | `t5-small`, 60,5 M paramètres, une mesure zero-shot et trois fine-tunages |
| Ablation sur la taille du corpus | Neuf entraînements, sous-ensembles emboîtés, budget d'optimisation identique pour les trois familles |
| Ablation d'architecture | Trois profondeurs du Transformer écrit à la main : 2, 4 et 6 couches |
| Métrique et analyse qualitative | ROUGE-1, ROUGE-2, ROUGE-L avec intervalles bootstrap à 95 %, plus une sélection d'exemples par run |

L'énoncé laisse le choix entre BLEU et ROUGE. La mesure retenue est ROUGE, orientée rappel et usuelle en résumé automatique, quand BLEU mesure une précision pensée pour la traduction.

> **État de la campagne.** Les douze expériences déclarées ont été exécutées et portent le statut `OK`. Elles cumulent 151 minutes de calcul sur une NVIDIA GeForce RTX 5060 Laptop GPU, sous la graine 42 et le corpus `00c0ee4e`.

## 1. Le corpus

Le corpus de travail est un tirage figé de 22 000 exemples de **CNN/DailyMail**, configuration `3.0.0`, sous graine 42 : 20 000 pour l'entraînement, 1 000 pour la validation, 1 000 pour le test. Le « 100 % » des ablations désigne ces 20 000 exemples, jamais les 287 113 exemples d'entraînement du corpus complet.

La configuration `3.0.0` est épinglée. Le hub publie trois versions sous le même identifiant et seule celle-ci laisse les entités nommées en clair ; ne pas la fixer laisserait un score changer sans qu'une ligne du dépôt bouge.

### Pourquoi un sous-ensemble, et pourquoi tiré

Le projet compare trois familles de modèles sur les mêmes données ; il ne cherche pas à égaler un score publié. La comparaison exige que les trois familles voient exactement le même corpus, elle n'exige pas que ce corpus soit tout ce qui est disponible.

Le corpus complet multiplierait par 14 le coût de chaque run, et la campagne des douze expériences avec lui. À 20 000 exemples, un modèle parti de zéro voit environ 9,9 millions de tokens source, soit à peu près 3 400 fois moins que ce que `t5-small` a vu en pré-entraînement ; à 287 113 exemples il en verrait 141 millions, soit 240 fois moins. Cela explique le choix budgétaire ; les écarts mesurés sur ce sous-ensemble sont présentés section 5.

Ce choix limite aussi la couverture lexicale : 76,0 % des identifiants utilisables du tokenizer apparaissent dans le corpus. Cela ne signifie toutefois pas que les autres lignes d'embedding ne reçoivent aucun gradient. La matrice est liée à la projection de sortie, et le softmax met à jour toutes les classes comme alternatives possibles. Les identifiants absents manquent surtout d'exemples positifs et d'usages directs dans l'entrée.

Le tirage, lui, remplace une troncature. Garder les 20 000 premiers exemples reprendrait l'ordre du fichier amont, dont rien ne garantit qu'il soit aléatoire. La graine casse cet ordre de façon rejouable : elle est écrite dans la configuration, et les empreintes du manifeste permettent de vérifier après coup qu'on a bien le même tirage.

Les trois splits reçoivent des graines dérivées, 42, 43 et 44, de sorte que changer `train_size` ne déplace ni la validation ni le test : porter l'entraînement à 30 000 exemples puis le ramener à 20 000 a laissé leurs deux empreintes identiques.

La validation ne relève aucun champ vide, aucun identifiant dupliqué, aucun résumé plus long que son document, et aucun document partagé entre les trois splits. Treize documents apparaissent deux fois dans l'entraînement, soit 0,065 % de la pondération. Ce sont des dépêches republiées, et les retirer serait un nettoyage silencieux du corpus de référence.

Les sous-ensembles d'ablation sont emboîtés, 10 % préfixe de 50 %, lui-même préfixe de 100 %. Tirés indépendamment, un écart entre deux points de la courbe mélangerait l'effet de la taille et celui de la composition de l'échantillon.

<!-- syntra:begin statistics -->
<!-- Généré par python -m src.data.fragments. Ne pas éditer à la main. -->

| Grandeur | Entraînement | Validation | Test |
| --- | --- | --- | --- |
| Mots par document, médiane | 634,0 | 617,0 | 637,5 |
| Mots par résumé, médiane | 49,0 | 54,0 | 52,0 |
| Tokens par document, moyenne | 988,5 | 965,5 | 1 000,2 |
| Tokens par document, médiane | 899,0 | 862,5 | 892,5 |
| Tokens par document, p95 | 1 961,1 | 1 946,8 | 2 006,1 |
| Tokens par résumé, médiane | 70,0 | 78,0 | 75,0 |
| Tokens par résumé, p95 | 130,0 | 145,1 | 133,1 |
| Taux de compression | 0,091 | 0,105 | 0,095 |
<!-- syntra:end statistics -->

### La tâche est extractive

Le taux de compression, 9 %, ressemble à celui d'un corpus de résumé extrême. C'est un faux ami : l'article est long et la référence l'est aussi, trois à quatre phrases pour 70 tokens médians. La grandeur qui dit la tâche est la part du résumé déjà présente dans l'article.

| Recouvrement résumé → article | Entraînement | Test |
| --- | --- | --- |
| Unigrammes | 84,5 % | 85,5 % |
| Bigrammes | 46,5 % | 48,6 % |
| Trigrammes | 27,4 % | 29,0 % |

Cinq mots sur six du résumé figurent dans le document, et près d'un bigramme sur deux. Sélectionner et recopier les bons fragments est donc une stratégie payante, ce qui change ce qu'on doit attendre du modèle from scratch : la copie n'est pas une impasse ici, elle est une part de la solution.

### La troncature

C'est le compromis le plus coûteux du projet.

| Plafond source | Documents coupés | Texte conservé | Coût relatif de l'encodeur |
| --- | --- | --- | --- |
| 256 | 98,6 % | 25,8 % | 0,25x |
| **512** | **85,3 %** | **50,0 %** | **1,0x** |
| 768 | 62,0 % | 69,1 % | 2,2x |
| 1 024 | 39,6 % | 82,2 % | 4,0x |
| 1 536 | 13,5 % | 94,9 % | 9,0x |

À 512 tokens, 85 % des articles sont coupés et l'encodeur ne voit que la moitié du texte source. L'attention coûte le carré de la longueur : passer à 1 024 pour récupérer 32 points multiplierait par quatre le coût de l'encodeur, ce que le budget d'une carte portable de 8 Go ne permet pas. C'est aussi le budget d'encodage de la littérature T5 sur ce corpus ; les travaux qui vont à 1 024 le font sur BART ou PEGASUS.

Les trois familles de modèles subissent exactement la même troncature, donc la comparaison reste équitable : ce qui est perdu est une part du plafond atteignable, pas l'équité. Et l'article de presse est écrit en pyramide inversée, l'essentiel d'abord, les puces de highlights suivant cet ordre : couper la queue coûte moins que « la moitié du texte » ne le laisse craindre.

Le plafond des cibles, 128 tokens, coupe 5,3 % des résumés et suit le p95 mesuré à 130. Un plafond de 64 tokens, celui qu'un corpus de résumé extrême autoriserait, en couperait 60 %.

## 2. Les trois familles comparées

La campagne oppose trois familles, et elles ne répondent pas à la même question.

| Famille | Ce que c'est | Paramètres |
| --- | --- | --- |
| `scratch_*` | Transformer encodeur-décodeur écrit bloc par bloc dans `src/models/scratch/` | 60 575 744 |
| `random_t5_*` | `T5ForConditionalGeneration` construit depuis la configuration épinglée, poids tirés au hasard | 60 506 624 |
| `pretrained_zero_shot`, `pretrained_ft_*` | la même classe, les mêmes formes, poids du hub | 60 506 624 |

**`random_t5_*` contre `pretrained_ft_*` isole le pré-entraînement.** Les deux branches ne partagent pas seulement un nombre de paramètres : ce sont les mêmes tenseurs, aux mêmes formes, dans le même graphe de calcul. Un écart de score ne peut donc venir que de la valeur initiale des poids. C'est le seul couple de ce dépôt qui autorise la phrase « cet écart est dû au pré-entraînement ».

**`scratch_*` contre `pretrained_ft_*` compare à budget de paramètres égal.** Le graphe, lui, diffère. Un écart y mélange le pré-entraînement et les choix d'architecture de T5, et la phrase que ce tableau autorise est plus faible : « à budget de paramètres et données égaux, le modèle pré-entraîné fait X de plus ».

Conséquence à anticiper avant de lire les résultats : le Transformer écrit à la main marquera probablement moins que `random_t5_*` alors que les deux pèsent pareil. Ce ne serait pas un défaut de câblage, mais l'effet des choix listés en 2.2. Les deux tableaux ne donneront donc pas le même « coût du départ aléatoire ».

### 2.1 Le Transformer écrit à la main

`torch.nn.Transformer` existe et fonctionne. L'objet du projet est de démontrer la compréhension de l'architecture, pas d'en consommer une implémentation. Chaque bloc est écrit séparément.

```text
src/models/scratch/
├── config.py                 hyperparamètres validés à la construction
├── embeddings.py             table partagée, mise à l'échelle par sqrt(d_model)
├── positional_encoding.py    encodage sinusoïdal fixe
├── attention.py              softmax(Q Kt / sqrt(d_k)) V
├── multi_head_attention.py   projections, découpage en têtes, concaténation
├── feed_forward.py           réseau position par position
├── masks.py                  masque de padding, masque causal
├── encoder_layer.py          self-attention + feed forward, résiduels
├── decoder_layer.py          self-attention masquée + cross-attention + FFN
├── encoder.py                pile d'encodeurs
├── decoder.py                pile de décodeurs + projection vers le vocabulaire
├── transformer.py            assemblage, teacher forcing, perte
└── generation.py             génération autorégressive, greedy et beam search
```

La forme retenue est celle de `t5-small`, pour que le budget de paramètres soit le même de part et d'autre :

```yaml
d_model: 512          d_ff: 2048           max_position: 512
num_heads: 8          dropout: 0.1         tie_embeddings: true
encoder_layers: 6     decoder_layers: 6    norm_first: true
```

| Bloc | Paramètres | Part |
| --- | --- | --- |
| Décodeur, 6 couches | 25 225 216 | 41,6 % |
| Encodeur, 6 couches | 18 915 328 | 31,2 % |
| Table d'embedding partagée | 16 435 200 | 27,1 % |
| **Total** | **60 575 744** | **100 %** |

### 2.2 Les 69 120 paramètres d'écart

Le Transformer écrit à la main pèse 0,11 % de plus que `t5-small`. L'écart se décompose entièrement, et chaque ligne est un choix de T5 que ce Transformer ne reprend pas :

| Origine | Paramètres |
| --- | --- |
| Biais des projections linéaires, que T5 n'a pas | + 67 584 |
| Biais des normalisations, que `T5LayerNorm` n'a pas | + 16 384 |
| Table d'embedding de T5, plus large de 28 lignes | − 14 336 |
| Table de biais de position relative de T5 | − 512 |
| **Total** | **+ 69 120** |

`t5-small` ne porte aucun biais : ni sur ses projections, ni sur ses normalisations. Sa table couvre 32 128 lignes quand le tokenizer n'en expose que 32 100, les 28 dernières restant inutilisées. Et sa notion de position est une table apprise de 32 seaux par tête, une par pile, là où l'encodage sinusoïdal du Transformer écrit à la main ne coûte aucun paramètre.

```text
t5-small@df1b051c
├── T5Config.from_pretrained(...) → T5ForConditionalGeneration(config)
│                                   → poids aléatoires        random_t5_*
└── T5ForConditionalGeneration.from_pretrained(...)
                                    → poids pré-entraînés sur C4
                                                              pretrained_*
```

| Élément architectural commun aux deux T5 | Valeur |
| --- | --- |
| Embedding partagé | 32 128 × 512 |
| Encodeur / décodeur | 6 blocs / 6 blocs |
| Attention | 8 têtes de 64 dimensions |
| Réseau feed-forward | 512 → 2 048 → 512, activation ReLU |
| Position | biais relatif à 32 seaux, pas d'encodage absolu ajouté aux embeddings |
| Normalisation | `T5LayerNorm`, de type RMS sans biais |
| Sortie | projection liée à la table d'embedding, 32 128 logits |
| Paramètres entraînables | 60 506 624 dans chaque branche |

Le carnet [03_transformer_walkthrough.ipynb](notebooks/03_transformer_walkthrough.ipynb) fait traverser un vrai batch à ces modèles et affiche la forme de chaque tenseur ; les sections qui suivent disent ce que ces formes recouvrent.

### L'attention

```text
écrit à la main  Attention(Q, K, V) = softmax(Q Kt / sqrt(d_k)) V
T5               Attention(Q, K, V) = softmax(Q Kt + biais)     V
```

`Q Kt` attribue à chaque requête un score contre chaque clé, le softmax en fait une distribution, et le produit avec `V` renvoie une moyenne pondérée des valeurs. Le découpage en têtes est une opération de forme, identique des deux côtés : le tenseur passe de `(2, seq, 512)` à `(2, 8, seq, 64)` par une vue et une transposition, l'attention s'applique tête par tête, puis le chemin inverse recompose `(2, seq, 512)` avant `W_O`.

**La division par `sqrt(d_k)` est là d'un côté et pas de l'autre.** Si les composantes de `Q` et `K` sont indépendantes, centrées et de variance unité, leur produit scalaire sur `d_k` dimensions a une variance de `d_k` : quand `d_k` grandit, les scores s'étalent, le softmax sature et son gradient s'annule. Le Transformer écrit à la main divise, comme l'article original. T5 obtient le même effet en absorbant le facteur dans l'initialisation de ses projections, et sa formule ne porte donc pas de division. C'est un détail que rien dans les formes ne signale, et qu'une réimplémentation « corrigée » de T5 ferait diverger du modèle du hub.

L'attention multi-têtes écrite à la main n'instancie pas `h` petites projections mais quatre projections larges, `W_Q`, `W_K`, `W_V` et `W_O`, toutes de `d_model` vers `d_model`, initialisées en Xavier uniforme avec des biais nuls. Ce sont ces biais qui pèsent les 67 584 paramètres de la section 2.2.

### La position

Les deux constructions s'y séparent nettement.

Le Transformer écrit à la main ajoute à l'embedding un encodage sinusoïdal fixe, la table étant mise à l'échelle par `sqrt(d_model)`. Rien n'est appris : la position est une fonction de l'indice, calculée une fois et gardée en tampon. Elle ne coûte aucun paramètre et elle est disponible dès le premier pas d'optimisation.

T5 n'ajoute aucun encodage positionnel à l'embedding, et ne met pas non plus la table à l'échelle. La position entre au niveau des scores, sous forme d'un biais appris par tête et par seau de distance relative, 32 seaux ici. Un seul bloc de chaque pile porte ce biais, les autres réutilisent le sien : la notion de position est apprise une fois pour l'encodeur et une fois pour le décodeur.

Trois conséquences. La longueur maximale de T5 n'est pas une propriété de son architecture, seulement de son budget mémoire, quand celle du Transformer écrit à la main est fixée par `max_position`. Le `random_t5_*` doit apprendre sa notion de position en même temps que le reste, alors que le pré-entraîné arrive avec. Et le Transformer écrit à la main n'a rien à apprendre de ce côté, ce qui joue en sa faveur sur un petit corpus sans compenser le reste.

### Les masques

Le masque de padding empêche le modèle de lire le remplissage ; sans lui, les prédictions dépendraient de la composition du lot. Le masque causal empêche la position `t` de voir les positions suivantes pendant le teacher forcing ; sans lui, le modèle lit la réponse qu'on lui demande de prédire, la perte s'effondre et la génération reste aléatoire.

Les deux sont additifs, dans les deux constructions : les positions interdites reçoivent la plus petite valeur finie du type, et non moins l'infini. Sur une séquence entièrement remplie de padding, moins l'infini donne une ligne de zéros divisée par zéro, donc des NaN. Un test couvre ce cas, qui ne se déclenche pas sur un lot ordinaire.

**Le décalage à droite.** L'entrée du décodeur est la cible décalée d'une position, préfixée du token de départ. Le tokenizer T5 n'a pas de token de début de séquence, donc l'identifiant de padding joue ce rôle. Passer `labels` au modèle suffit : il construit ce décalage lui-même et y remet du padding là où les labels portent `-100`.

### Pre-norm et normalisation

```text
post-norm  x = LayerNorm(x + Sublayer(x))     article original
pre-norm   x = x + Sublayer(LayerNorm(x))     écrit à la main, défaut du projet
pre-norm   x = x + Sublayer(T5LayerNorm(x))   T5
```

Le pre-norm laisse le chemin résiduel libre de toute normalisation, donc le gradient atteint la première couche sans distorsion, et il s'entraîne sans le long warmup que le post-norm réclame. Les deux constructions le retiennent ; le post-norm reste accessible côté Transformer écrit à la main par `norm_first: false`, et les deux dispositions sont couvertes par les tests.

La normalisation elle-même diffère. Le Transformer écrit à la main utilise `nn.LayerNorm`, qui recentre puis met à l'échelle, avec un gain et un biais. `T5LayerNorm` est une normalisation RMS : elle divise par la racine de la moyenne des carrés, sans recentrer et sans biais additif. Ce sont les 16 384 paramètres de la section 2.2.

### Poids liés et couverture du vocabulaire

Les trois familles attachent leurs poids : une seule table sert l'encodeur, le décodeur et la projection de sortie, les trois tenseurs étant le même objet en mémoire. L'attachement économise `vocab_size * d_model`, soit 16 449 536 paramètres côté T5 et 16 435 200 côté Transformer écrit à la main, et force la vue d'entrée et la vue de sortie d'un token à s'accorder. Toutes multiplient l'état par `d_model ** -0.5` avant la projection, pour compenser l'échelle de la table qu'elle réutilise.

Sur les 32 100 identifiants utilisables du tokenizer, 24 480 apparaissent au moins une fois dans les 21,3 millions d'occurrences du corpus d'entraînement, soit 76 %. Les autres n'ont aucun exemple direct comme token d'entrée ou cible positive. Ils ne sont pourtant pas des « paramètres morts » : parce que la table est aussi la projection de sortie, le softmax leur transmet un gradient comme classes concurrentes.

La concentration aggrave le constat : 50 % des occurrences tiennent dans 92 types, et il faut 15 863 types pour couvrir 99 % du texte. Les 2 326 types vus moins de dix fois ont une ligne d'embedding mise à jour trop peu souvent pour valoir mieux que du bruit. La table pèse 27,2 % de chaque modèle, et c'est le bloc que le pré-entraînement livre déjà appris. Un tokenizer réduit au corpus libérerait plusieurs millions de paramètres mais casserait le partage entre les trois familles, donc l'équité de la comparaison.

### Ce que les tests garantissent

Certaines propriétés ne se voient pas sur une courbe de perte, et ce sont celles que la comparaison suppose.

**Le décodeur écrit à la main ne lit pas le futur.** Modifier le token de l'entrée décodeur à la position `k` laisse les logits des positions antérieures strictement inchangés, et modifie ceux de la position `k`. La seconde moitié de l'assertion compte autant que la première, sans quoi un modèle qui ignorerait entièrement son entrée passerait le test.

**Le padding ne change rien.** Ajouter du remplissage à la source laisse les logits identiques, et un masque mal diffusé fait échouer ce test alors que la perte continue de descendre.

**Les deux T5 sont la même architecture.** `test_random_initialisation_keeps_the_exact_t5_architecture` compare la classe, la configuration complète, le nombre de paramètres et la forme de chaque entrée du `state_dict` entre la branche aléatoire et la branche chargée. Une divergence, même d'un tenseur, y échoue.

**Les fichiers `random_t5_*` et `pretrained_ft_*` ne diffèrent que par l'initialisation.** `test_compared_models_only_differ_by_initialisation` lit les fichiers réels sous `configs/experiments/` et vérifie, pour chacune des trois proportions, que la baseline, l'identifiant, la révision, le bloc `dataset`, le bloc `training` et le bloc `evaluation` sont identiques de part et d'autre.

**Les trois familles s'entraînent sous un seul budget.** `test_the_three_families_train_on_one_budget` refuse qu'un fichier déclare autre chose que 3 époques à 1e-4. Une famille qui garderait un budget à elle transformerait chaque écart entre familles en écart entre budgets.

**Un score ne peut pas être classé sous la mauvaise initialisation.** `describe()` porte `initialization` et `mode`, écrits dans l'enregistrement du run, et `test_random_initialisation_is_reported_as_from_scratch` fixe ce que la branche aléatoire y déclare.

S'y ajoutent la vérification de la formule d'attention sur une entrée construite à la main, l'absence de NaN sur une ligne entièrement masquée, la reproduction de l'encodage positionnel publié position par position, l'économie exacte de `vocab_size * d_model` paramètres par l'attachement, le surapprentissage d'un lot unique jusqu'à moins de 20 % de la perte initiale, et du côté du décodage le retrait du token de départ que l'encodeur-décodeur rend en position zéro, le respect du budget `max_new_tokens`, le retour à un résumé par document en faisceau et l'absence de gradient pendant la génération. Le carnet 03 vérifie qu'un bloc d'attention de T5 recalculé à la main redonne exactement les poids que le modèle produit, et que la perte d'un modèle non entraîné se tient autour de `ln(32 128)`.

## 3. La baseline pré-entraînée

`t5-small`, 60 506 624 paramètres, est mesuré en zero-shot, sans aucun entraînement et avec le préfixe de tâche `summarize:`, puis fine-tuné sur 10 %, 50 % et 100 % du même corpus. Pour chaque proportion, son jumeau initialisé aléatoirement reçoit exactement les mêmes hyperparamètres d'entraînement et de décodage, et le Transformer écrit à la main les reçoit aussi.

Les trois familles partagent le tokenizer, le corpus, le chargeur, la boucle d'entraînement, le décodage et la métrique. Le modèle zero-shot saute l'étape d'entraînement parce que ses poids ne bougent pas, pas parce qu'il emprunte un autre chemin.

Les sept mesures fondées sur `t5-small`, aléatoires ou pré-entraînées, utilisent la révision `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`. Les poids fine-tunés sont les nôtres, mais l'architecture dans laquelle ils sont chargés vient du hub, et sans révision fixée elle peut changer sans que rien ne change dans le dépôt.

## 4. Protocole d'évaluation

Le bloc d'évaluation est identique dans les douze fichiers d'expérience : `num_beams` 4, `max_new_tokens` 128, `no_repeat_ngram_size` 3, 1 000 rééchantillonnages bootstrap à 95 %. Le budget de décodage suit le plafond des cibles, lui-même posé sur le p95 des références mesuré section 1.

**ROUGE-L, pas ROUGE-Lsum.** `rougeLsum` découpe la référence sur ses sauts de ligne et apparie chaque phrase séparément ; ROUGE-L exige une seule sous-séquence traversant toute la paire. Sur une référence de trois à quatre phrases, la seconde mesure est nettement plus sévère, et les chiffres publiés sur CNN/DailyMail sont des ROUGE-Lsum. **Aucun chiffre de ce rapport ne s'y compare.** Toutes les comparaisons faites ici sont internes : même métrique, même jeu de test, pour tous les modèles.

Une prédiction vide vaut zéro et reste dans la moyenne. Retirer les documents sur lesquels un modèle a échoué relèverait sa moyenne pour avoir échoué. Le compte des prédictions vides est reporté à côté du score, ce qui sépare un score faible d'un modèle cassé.

Chaque score porte un intervalle bootstrap à 95 %, calculé sous la graine du run. Sans lui, un écart de deux millièmes se lit comme un classement.

Seul un run complet remplit une colonne de score. Un run partiel garde sa ligne et son statut, ses cellules de score restent vides.

Des runs mesurés différemment ne sont pas mis dans un même tableau ni sur une même figure. Le budget de décodage, la largeur de faisceau et les réglages de ROUGE voyagent dans chaque enregistrement, et l'agrégation refuse d'écrire si deux runs d'une même étude divergent.

## 5. Résultats

### Zero-shot, puis fine-tuné

<!-- syntra:begin dataset_size -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Variante | Corpus | Exemples | ROUGE-L | IC 95 % |
| --- | --- | --- | --- | --- |
| `pretrained_ft` | 10 % | 2 000 | 0,2867 | [0,2784, 0,2945] |
| `pretrained_ft` | 50 % | 10 000 | 0,2894 | [0,2818, 0,2973] |
| `pretrained_ft` | 100 % | 20 000 | 0,2915 | [0,2836, 0,2997] |
| `random_t5` | 10 % | 2 000 | 0,0771 | [0,0745, 0,0798] |
| `random_t5` | 50 % | 10 000 | 0,0985 | [0,0961, 0,1011] |
| `random_t5` | 100 % | 20 000 | 0,1172 | [0,1145, 0,1200] |
| `scratch` | 10 % | 2 000 | 0,0343 | [0,0327, 0,0357] |
| `scratch` | 50 % | 10 000 | 0,1014 | [0,0987, 0,1042] |
| `scratch` | 100 % | 20 000 | 0,1156 | [0,1129, 0,1184] |
| `pretrained_zero_shot` | sans objet | sans objet | 0,2751 | [0,2672, 0,2829] |
<!-- syntra:end dataset_size -->

`t5-small` obtient 0,2751 en zero-shot et 0,2915 après fine-tuning sur les 20 000 exemples : le fine-tunage sur ce corpus lui rapporte 0,0164 de ROUGE-L, et le passage de 2 000 à 20 000 exemples 0,0049. Sur la même décade de données, `random_t5` gagne 0,0401 et le Transformer from scratch 0,0813, mais ils restent respectivement à 0,1744 et 0,1759 du modèle pré-entraîné à 100 % du corpus.

### Performance contre taille du corpus d'entraînement

![Performance contre taille du corpus](reports/figures/performance_vs_dataset_size.png)

La courbe pré-entraînée est presque plate : multiplier le corpus par dix ne rapporte que 0,0049 de ROUGE-L, un écart inférieur à la largeur des intervalles de confiance. Les deux courbes parties de zéro montent beaucoup plus, puisqu'elles apprennent simultanément la langue et la tâche, sans toutefois rejoindre le modèle pré-entraîné sur la plage mesurée.

### Le coût du départ aléatoire, à architecture identique

<!-- syntra:begin initialisation -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Proportion | Exemples | `t5-small` aléatoire | `t5-small` fine-tuné | Écart absolu | Écart relatif |
| --- | --- | --- | --- | --- | --- |
| 10 % | 2 000 | 0,0771 | 0,2867 | 0,2095 | +272 % |
| 50 % | 10 000 | 0,0985 | 0,2894 | 0,1909 | +194 % |
| 100 % | 20 000 | 0,1172 | 0,2915 | 0,1744 | +149 % |
<!-- syntra:end initialisation -->

Ce tableau est le seul du rapport dont l'écart s'attribue au pré-entraînement seul : les deux colonnes portent les mêmes tenseurs, aux mêmes formes, dans le même graphe, et ne diffèrent que par la valeur initiale des poids.

### Le coût du départ aléatoire, à budget de paramètres égal

<!-- syntra:begin families -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Proportion | Exemples | Transformer from scratch | `t5-small` fine-tuné | Écart absolu | Écart relatif |
| --- | --- | --- | --- | --- | --- |
| 10 % | 2 000 | 0,0343 | 0,2867 | 0,2524 | +736 % |
| 50 % | 10 000 | 0,1014 | 0,2894 | 0,1880 | +185 % |
| 100 % | 20 000 | 0,1156 | 0,2915 | 0,1759 | +152 % |
<!-- syntra:end families -->

Ici les deux colonnes pèsent le même nombre de paramètres à 0,11 % près, mais ne sont pas la même architecture. L'écart y mélange le pré-entraînement et les choix listés en 2.2, et la lecture qu'il autorise s'arrête à « à budget de paramètres et données égaux, le modèle pré-entraîné fait X de plus ».

### À partir de quelle taille le from scratch devient-il compétitif ?

La campagne fournit trois points par famille. Pour le Transformer écrit à la main, l'écart au T5 pré-entraîné se referme de 0,0765 quand les données sont multipliées par dix ; il reste 0,1759 à combler à 20 000 exemples. Une extrapolation log-linéaire placerait l'égalité vers quatre millions d'exemples. Pour `random_t5`, dont l'architecture est identique à celle du pré-entraîné, la fermeture n'est que de 0,0353 par décade et repousserait l'égalité vers l'ordre du milliard d'exemples.

Ces ordres de grandeur ne sont pas des prévisions : trois points mesurés sur une seule décade ne justifient pas une extrapolation de deux à cinq décades, et les pentes varient déjà à l'intérieur de la plage observée. Ils indiquent seulement que le corpus complet de CNN/DailyMail, 287 113 exemples, resterait très probablement insuffisant pour combler l'écart sous ce protocole.

### Ablation d'architecture

Une seconde ablation fait varier la profondeur du Transformer écrit à la main, tout le reste étant identique, corpus compris. Trois points, 2, 4 et 6 couches, suffisent à distinguer une profondeur qui aide d'une profondeur qui sature ; `scratch_100` fournit le troisième, ce qui limite l'étude à deux entraînements supplémentaires.

<!-- syntra:begin architecture -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Expérience | Couches | Paramètres | ROUGE-L | IC 95 % | Entraînement |
| --- | --- | --- | --- | --- | --- |
| `scratch_100_layers2` | 2 + 2 | 31 150 080 | 0,1067 | [0,1040, 0,1094] | 441 s |
| `scratch_100_layers4` | 4 + 4 | 45 862 912 | 0,1095 | [0,1067, 0,1125] | 722 s |
| `scratch_100` | 6 + 6 | 60 575 744 | 0,1156 | [0,1129, 0,1184] | 900 s |
<!-- syntra:end architecture -->

Le temps d'entraînement est reporté à côté du score : une profondeur qui coûte plus et ne rapporte rien est un résultat, et sans la colonne de coût le tableau n'en montre que la moitié.

## 6. Analyse qualitative

Chaque run sélectionne son meilleur cas, son pire cas et un tirage aléatoire, de façon déterministe. Le tirage compte autant que les extrêmes, qui sont par construction les deux exemples les moins représentatifs. Les douze runs disposent de prédictions et participent donc à l'analyse qualitative.

Le même article résumé par les deux (un mur de traces de dinosaures en Bolivie, document `dec77c4f`) :

> **Référence.** There are 462 trails of 5,055 prints on a vertical limestone slab in Bolivia . Rock was pushed upwards by tectonic movement, standing 390 feet tall . Site's thought to be the largest dinosaur trackway in the world […]

> **`t5-small` zero-shot, 0,208.** the attraction, at Fancesa limestone quarry in Sucre, is the result of tectonic activity forcing earth upwards. it comprises some 462 trails made up of 5,055 prints - and frequent landslides reveal new ones […]

> **`t5-small` fine-tuné, 100 %, 0,254.** The attraction, at Fancesa limestone quarry in Sucre, Bolivia, comprises some 462 trails made up of 5,055 prints. The wall, which is the largest dinosaur trackway in the world, is approximately 390 feet (120 metres) tall […]

Le zero-shot reprend de vraies phrases de l'article, en minuscules et sans les resserrer. Le fine-tuné dit la même chose, capitalisée, raccourcie et recousue en phrases complètes. Le contenu retenu est presque le même : ce sont la forme et la ponctuation qui changent.

**Le fine-tuning apprend le format, pas le contenu.** Le tableau ci-dessous compte les prédictions commençant par une minuscule : 71 % en zero-shot, moins de 1 % dès le premier fine-tunage. Le contenu, lui, ne bouge pas, et c'est ce qui explique que ROUGE ne gagne que 0,016 entre les deux.

<!-- syntra:begin capitalisation -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Modèle | Prédictions commençant par une minuscule |
| --- | --- |
| `pretrained_ft_10` | 0,8 % |
| `pretrained_ft_50` | 0,5 % |
| `pretrained_ft_100` | 0,2 % |
| `random_t5_10` | 0,0 % |
| `random_t5_50` | 0,0 % |
| `random_t5_100` | 0,1 % |
| `scratch_10` | 0,0 % |
| `scratch_50` | 77,5 % |
| `scratch_100` | 94,9 % |
| `scratch_100_layers2` | 96,7 % |
| `scratch_100_layers4` | 99,7 % |
| `pretrained_zero_shot` | 71,0 % |
| Références | 0,0 % |
<!-- syntra:end capitalisation -->

La capitalisation seule ne mesure pas la qualité. `random_t5_*` commence presque toujours par une majuscule tout en restant très loin du score pré-entraîné ; inversement, les variantes `scratch_*` les plus profondes produisent majoritairement une minuscule. Ce tableau caractérise donc le format appris, en complément de ROUGE et de la longueur des sorties.

## 7. Traçage et magasin de modèles

Le magasin MLflow est une base **PostgreSQL 18** dédiée, `syntra_mlflow`, sur un serveur du réseau local. Elle porte un rôle propre, propriétaire de la base et sans droit ailleurs : le serveur héberge d'autres applications, et un magasin d'expériences n'a pas à s'y connecter en superutilisateur.

L'URI de connexion porte un mot de passe et ne vit donc pas dans le dépôt. `.env.example` en donne la forme, `.env` la valeur, et `.gitignore` garde le second dehors. `src/tracking/store.py` résout l'URI depuis l'environnement puis depuis ce fichier, dans cet ordre, et le lanceur affiche le magasin qu'il a obtenu, mot de passe masqué, avant la première expérience : une campagne qui aurait tracé dans un fichier local au lieu de la base partagée le dit à la première seconde.

**La base ne porte aucun poids.** Elle porte les métadonnées des runs : paramètres, métriques, tags, et un pointeur vers les artefacts. Ceux-ci sont des fichiers, écrits sous `MLFLOW_ARTIFACT_ROOT`.

**Les modèles mesurés sont déposés dans le magasin.** Les deux branches T5 passent par `mlflow.transformers`, qui embarque le tokenizer et la configuration de génération avec les poids ; le Transformer écrit à la main passe par `mlflow.pytorch`, qui dépose son `state_dict`. Chaque modèle est enregistré au registre sous `syntra-<expérience>` et se recharge par ce nom.

- Les poids déposés sont ceux qui ont été **évalués**, relus depuis le meilleur checkpoint, et non l'objet en fin d'entraînement. Quand l'arrêt anticipé a retenu une époque antérieure, les deux diffèrent, et déposer le second stockerait des poids que personne n'a mesurés.
- Seul un run `OK` entre au registre. Un run `PARTIAL` a vu deux pas d'optimisation et huit documents de test : ses poids existent et ne veulent rien dire.
- Un répertoire de run laissé par une autre architecture que celle déclarée aujourd'hui porte le statut `STALE_CONFIG` : il n'entre ni dans un tableau ni au registre, plutôt que de fournir un score sous un nom qui ne lui correspond plus.

Un échec de traçage n'interrompt jamais un run. Le résultat d'une expérience est l'enregistrement sur disque ; le magasin en est le miroir, et `python -m src.tracking.log --all` le reconstruit après coup.

## 8. Limites

Le corpus est un tirage de 20 000 exemples, pas CNN/DailyMail complet, qui en compte 287 113. Toutes les conclusions valent sur cette plage.

**La troncature à 512 tokens écarte la moitié du texte source et touche 85 % des articles.** C'est la limite dominante du projet sur ce corpus. Elle s'applique aux trois familles également, donc elle ne biaise pas la comparaison, mais elle abaisse le plafond atteignable par toutes.

**Un seul des deux tableaux de comparaison isole le pré-entraînement.** `random_t5_*` et `pretrained_ft_*` sont la même architecture, la même révision, les mêmes 60 506 624 paramètres, le même tokenizer, les mêmes données et les mêmes hyperparamètres : leur seule variable est la provenance des poids initiaux. Le Transformer écrit à la main partage le budget de paramètres et tout le reste du protocole, mais pas le graphe, et l'écart qu'il donne mélange deux causes.

**Le budget d'optimisation commun sous-entraîne les modèles partis de zéro.** Trois époques à 1e-4 sont taillées pour un fine-tunage. Un modèle sans poids pré-entraînés en tirerait davantage avec un budget plus long et un pas plus grand, et le lui accorder rendrait les trois tableaux incomparables. Ce que la campagne mesure est donc l'écart à budget d'optimisation constant, ce qui fait partie de la réponse plutôt que de la fausser.

Chaque configuration n'est entraînée que sous une graine, 42. Les intervalles publiés sont des intervalles bootstrap sur les 1 000 documents de test : ils mesurent l'échantillonnage du jeu d'évaluation, pas la variance d'entraînement.

**Les extrapolations de taille de corpus restent exploratoires.** Elles reposent sur trois proportions emboîtées, une seule graine d'entraînement et une seule décade de données observée. La campagne mesure solidement les écarts sur 2 000, 10 000 et 20 000 exemples ; elle ne démontre pas à quelle taille exacte les courbes se croiseraient.

## Pour reproduire

```bash
make data                                     # corpus figé, sous graine 42
make reproduce MODE=full                      # les douze expériences, les tableaux, les figures
```

Étape par étape :

```bash
python -m src.experiments.run --all           # les douze expériences
make ablation                                 # les ablations taille de corpus et architecture
make figures                                  # les quatre figures
make report-sync                              # les tableaux de ce rapport
make corpus-sync                              # les tableaux du corpus
```

Les tableaux de ce rapport se relisent dans `reports/results/experiments.csv`.

Le magasin demande un `.env` rempli sur le modèle de `.env.example` :

```bash
make mlflow-ui                                # les runs et les modèles, port 5000
python -m src.tracking.log --all              # rejoue les enregistrements vers le magasin
```
