# Rapport

Entraîner un Transformer encodeur-décodeur pour le résumé automatique, puis le comparer à un modèle pré-entraîné de référence sur les mêmes données de test.

| Exigence | Ce qui a été fait |
| --- | --- |
| Transformer encodeur-décodeur from scratch | Écrit composant par composant en PyTorch, 15,6 M paramètres, entraîné sur 10 %, 50 % et 100 % du corpus |
| Modèle pré-entraîné, zero-shot puis fine-tuné | `t5-small`, 60,5 M paramètres, une mesure zero-shot et trois fine-tunages |
| Ablation sur la taille du corpus | Sept runs, sous-ensembles emboîtés, budget d'époques constant |
| Ablation d'architecture | Trois profondeurs, 2, 4 et 6 couches, tout le reste identique |
| Métrique et analyse qualitative | ROUGE-1, ROUGE-2, ROUGE-L avec intervalles bootstrap à 95 %, plus une sélection d'exemples par run |

L'énoncé laisse le choix entre BLEU et ROUGE. La mesure retenue est ROUGE, orientée rappel et usuelle en résumé automatique, quand BLEU mesure une précision pensée pour la traduction.

> **État.** Le corpus de travail est construit et figé, `dataset_version = 00c0ee4e`. Les sections 1 à 4 et 7 sont écrites sur des grandeurs mesurées. Les sections 5 et 6 portent `NOT_RUN` : leurs tableaux sont générés depuis les enregistrements de runs et restent vides tant que la campagne n'a pas écrit.

## 1. Le corpus

Le corpus de travail est un tirage figé de 22 000 exemples de **CNN/DailyMail**, configuration `3.0.0`, sous graine 42 : 20 000 pour l'entraînement, 1 000 pour la validation, 1 000 pour le test. Le « 100 % » des ablations désigne ces 20 000 exemples, jamais les 287 113 exemples d'entraînement du corpus complet.

La configuration `3.0.0` est épinglée. Le hub publie trois versions sous le même identifiant et seule celle-ci laisse les entités nommées en clair ; ne pas la fixer laisserait un score changer sans qu'une ligne du dépôt bouge.

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

Deux choses tempèrent le chiffre. Les deux familles de modèles subissent exactement la même troncature, donc la comparaison reste équitable : ce qui est perdu est une part du plafond atteignable, pas l'équité. Et l'article de presse est écrit en pyramide inversée, l'essentiel d'abord, les puces de highlights suivant cet ordre : couper la queue coûte moins que « la moitié du texte » ne le laisse craindre.

Le plafond des cibles, 128 tokens, coupe 5,3 % des résumés et suit le p95 mesuré à 130. Un plafond de 64 tokens, celui qu'un corpus de résumé extrême autoriserait, en couperait 60 %.

## 2. Le Transformer from scratch

`torch.nn.Transformer` existe et fonctionne. L'objet du projet est de démontrer la compréhension de l'architecture, pas d'en consommer une implémentation. Chaque bloc est écrit séparément, et chaque décision de conception est couverte par un test qui échouerait si elle était fausse.

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

### L'attention

```text
Attention(Q, K, V) = softmax(Q Kt / sqrt(d_k)) V
```

`Q Kt` attribue à chaque requête un score contre chaque clé, le softmax en fait une distribution, et le produit avec `V` renvoie une moyenne pondérée des valeurs.

La division par `sqrt(d_k)` n'est pas cosmétique. Si les composantes de `Q` et `K` sont indépendantes, centrées et de variance unité, leur produit scalaire sur `d_k` dimensions a une variance de `d_k`. Quand `d_k` grandit, les scores s'étalent, le softmax sature et son gradient s'annule. La division ramène la variance à un.

L'attention multi-têtes n'instancie pas `h` petites projections mais quatre projections larges, `W_Q`, `W_K`, `W_V` et `W_O`, toutes de `d_model` vers `d_model`, initialisées en Xavier uniforme avec des biais nuls. Le découpage en têtes est une opération de forme : le tenseur passe de `(2, seq, 256)` à `(2, 8, seq, 32)` par une vue et une transposition, l'attention s'applique tête par tête, puis le chemin inverse recompose `(2, seq, 256)` avant `W_O`.

### Les masques

Le masque de padding empêche le modèle de lire le remplissage ; sans lui, les prédictions dépendraient de la composition du lot. Le masque causal empêche la position `t` de voir les positions suivantes pendant le teacher forcing ; sans lui, le modèle lit la réponse qu'on lui demande de prédire, la perte s'effondre et la génération reste aléatoire.

Les positions interdites reçoivent la plus petite valeur finie du type, et non moins l'infini. Sur une séquence entièrement remplie de padding, moins l'infini donne une ligne de zéros divisée par zéro, donc des NaN. Un test couvre ce cas, qui ne se déclenche pas sur un lot ordinaire.

### Pre-norm plutôt que post-norm

```text
post-norm  x = LayerNorm(x + Sublayer(x))     article original
pre-norm   x = x + Sublayer(LayerNorm(x))     défaut du projet
```

Le pre-norm laisse le chemin résiduel libre de toute normalisation, donc le gradient atteint la première couche sans distorsion, et il s'entraîne sans le long warmup que le post-norm réclame. Le post-norm reste accessible par `norm_first: false`, et les deux dispositions sont couvertes par les tests.

### La configuration retenue et son coût

```yaml
d_model: 256          d_ff: 1024           max_position: 512
num_heads: 8          dropout: 0.1         tie_embeddings: true
encoder_layers: 4     decoder_layers: 4    norm_first: true
```

| Bloc | Paramètres | Part |
| --- | --- | --- |
| Table d'embedding partagée | 8 217 600 | 52,7 % |
| Décodeur, 4 couches | 4 214 272 | 27,0 % |
| Encodeur, 4 couches | 3 159 552 | 20,3 % |
| **Total** | **15 591 424** | **100 %** |

Une seule table sert l'encodeur, le décodeur et la projection de sortie : `encoder.embedding` et `decoder.embedding` sont le même objet en mémoire, et `decoder.output_projection.weight is embedding.weight` vaut `True`. L'attachement économise `vocab_size * d_model` paramètres, 15 591 424 au lieu de 23 809 024, et force la vue d'entrée et la vue de sortie d'un token à s'accorder.

### Le vocabulaire décide plus que la profondeur

Sur les 32 100 entrées du tokenizer, 24 480 apparaissent au moins une fois dans les 21,3 millions d'occurrences du corpus d'entraînement, soit 76 %. Les 7 620 restantes représentent 1 950 720 paramètres, **12,5 % du modèle, qui ne reçoivent jamais le moindre gradient** : initialisés au hasard, sauvegardés dans chaque checkpoint, ils n'apprennent rien.

La concentration aggrave le constat : 50 % des occurrences tiennent dans 92 types, et il faut 15 863 types pour couvrir 99 % du texte. Les 2 326 types vus moins de dix fois ont une ligne d'embedding mise à jour trop peu souvent pour valoir mieux que du bruit.

La table pèse 69 % du modèle à 2 couches, 52,7 % à 4 et 42,6 % à 6. Faire varier la profondeur ne fait donc varier qu'une minorité des paramètres pendant que la majorité reste sous-entraînée. Un tokenizer réduit au corpus libérerait plusieurs millions de paramètres mais casserait le partage avec la baseline, donc l'équité de la comparaison : c'est un compromis explicite.

### Ce que les tests garantissent

Deux propriétés ne se voient pas sur une courbe de perte. Le décodeur ne lit pas le futur : modifier le token de l'entrée décodeur à la position `k` laisse les logits des positions antérieures strictement inchangés, et modifie ceux de la position `k` ; la seconde moitié de l'assertion compte autant que la première, sans quoi un modèle qui ignorerait entièrement son entrée passerait le test. Le padding ne change rien : ajouter du remplissage à la source laisse les logits identiques, et un masque mal diffusé fait échouer ce test alors que la perte continue de descendre.

S'y ajoutent la vérification de la formule d'attention sur une entrée construite à la main, l'absence de NaN sur une ligne entièrement masquée, la reproduction de l'encodage positionnel publié position par position, l'économie exacte de `vocab_size * d_model` paramètres par l'attachement, et le surapprentissage d'un lot unique jusqu'à moins de 20 % de la perte initiale. À l'initialisation, la perte vaut environ 10,9 sur un lot aléatoire contre 10,4 attendus pour une distribution uniforme sur 32 100 classes : une valeur très éloignée signale un câblage défectueux avant le premier entraînement.

## 3. La baseline pré-entraînée

`t5-small`, 60 506 624 paramètres, soit 3,9 fois le modèle from scratch. Elle est mesurée deux fois : en zero-shot, sans aucun entraînement, avec le préfixe de tâche `summarize:` ; puis fine-tunée sur 10 %, 50 % et 100 % du même corpus, avec le même budget d'époques et le même décodage que le modèle from scratch.

Les deux familles partagent le tokenizer, le corpus, le chargeur, la boucle d'entraînement, le décodage et la métrique. Le modèle zero-shot saute l'étape d'entraînement parce que ses poids ne bougent pas, pas parce qu'il emprunte un autre chemin.

Les quatre mesures sont prises sous la révision `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` de `t5-small`. Les poids fine-tunés sont les nôtres, mais l'architecture dans laquelle ils sont chargés vient du hub, et sans révision fixée elle peut changer sans que rien ne change dans le dépôt.

## 4. Protocole d'évaluation

Le bloc d'évaluation est identique dans les neuf fichiers d'expérience : `num_beams` 4, `max_new_tokens` 128, `no_repeat_ngram_size` 3, 1 000 rééchantillonnages bootstrap à 95 %. Le budget de décodage suit le plafond des cibles, lui-même posé sur le p95 des références mesuré section 1.

**ROUGE-L, pas ROUGE-Lsum.** `rougeLsum` découpe la référence sur ses sauts de ligne et apparie chaque phrase séparément ; ROUGE-L exige une seule sous-séquence traversant toute la paire. Sur une référence de trois à quatre phrases, la seconde mesure est nettement plus sévère, et les chiffres publiés sur CNN/DailyMail sont des ROUGE-Lsum. **Aucun chiffre de ce rapport ne s'y compare.** Toutes les comparaisons faites ici sont internes : même métrique, même jeu de test, pour tous les modèles.

Quatre règles encadrent la mesure.

Une prédiction vide vaut zéro et reste dans la moyenne. Retirer les documents sur lesquels un modèle a échoué relèverait sa moyenne pour avoir échoué. Le compte des prédictions vides est reporté à côté du score, ce qui sépare un score faible d'un modèle cassé.

Chaque score porte un intervalle bootstrap à 95 %, calculé sous la graine du run. Sans lui, un écart de deux millièmes se lit comme un classement.

Seul un run complet remplit une colonne de score. Un run partiel garde sa ligne et son statut, ses cellules de score restent vides.

Des runs mesurés différemment ne sont pas mis dans un même tableau ni sur une même figure. Le budget de décodage, la largeur de faisceau et les réglages de ROUGE voyagent dans chaque enregistrement, et l'agrégation refuse d'écrire si deux runs d'une même étude divergent.

## 5. Résultats

> `NOT_RUN`. Les tableaux ci-dessous sont générés par `make report-sync` depuis les enregistrements de runs. Les lectures seront écrites sur les mesures, pas avant.

### Performance contre taille du corpus d'entraînement

![Performance selon la taille du corpus](reports/figures/performance_vs_dataset_size.png)

<!-- syntra:begin families -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Proportion | Exemples | from scratch | `t5-small` fine-tuné | Écart absolu | Écart relatif | Statut |
| --- | --- | --- | --- | --- | --- | --- |
| 10 % |  |  |  |  |  | `scratch_10` `NOT_RUN`, `pretrained_ft_10` `NOT_RUN` |
| 50 % |  |  |  |  |  | `scratch_50` `NOT_RUN`, `pretrained_ft_50` `NOT_RUN` |
| 100 % |  |  |  |  |  | `scratch_100` `NOT_RUN`, `pretrained_ft_100` `NOT_RUN` |
<!-- syntra:end families -->

### Ablation d'architecture

Une seconde ablation fait varier la profondeur, tout le reste étant identique, corpus compris.

<!-- syntra:begin architecture -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Expérience | Couches | Paramètres | ROUGE-L | IC 95 % | Entraînement | Statut |
| --- | --- | --- | --- | --- | --- | --- |
| `scratch_100_layers2` | 2 + 2 |  |  |  |  | `NOT_RUN` |
| `scratch_100` | 4 + 4 |  |  |  |  | `NOT_RUN` |
| `scratch_100_layers6` | 6 + 6 |  |  |  |  | `NOT_RUN` |
<!-- syntra:end architecture -->

## 6. Analyse qualitative

Chaque run sélectionne son meilleur cas, son pire cas et un tirage aléatoire, de façon déterministe. Le tirage compte autant que les extrêmes, qui sont par construction les deux exemples les moins représentatifs.

> `NOT_RUN`. Les exemples seront pris dans les enregistrements de la campagne.

<!-- syntra:begin capitalisation -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

Aucun run complet n'a écrit de prédictions : la table est vide.
<!-- syntra:end capitalisation -->

## 7. Traçage et magasin de modèles

Le magasin MLflow est une base **PostgreSQL 18** dédiée, `syntra_mlflow`, sur un serveur du réseau local. Elle porte un rôle propre, propriétaire de la base et sans droit ailleurs : le serveur héberge d'autres applications, et un magasin d'expériences n'a pas à s'y connecter en superutilisateur.

L'URI de connexion porte un mot de passe et ne vit donc pas dans le dépôt. `.env.example` en donne la forme, `.env` la valeur, et `.gitignore` garde le second dehors. `src/tracking/store.py` résout l'URI depuis l'environnement puis depuis ce fichier, dans cet ordre, et le lanceur affiche le magasin qu'il a obtenu, mot de passe masqué, avant la première expérience : une campagne qui aurait tracé dans un fichier local au lieu de la base partagée le dit à la première seconde.

**La base ne porte aucun poids.** Elle porte les métadonnées des runs : paramètres, métriques, tags, et un pointeur vers les artefacts. Ceux-ci sont des fichiers, écrits sous `MLFLOW_ARTIFACT_ROOT`.

**Les modèles mesurés sont déposés dans le magasin.** Un modèle par run complet, dans la saveur qui lui correspond : `mlflow.transformers` pour `t5-small`, qui fait le tour du tokenizer et de la configuration de génération avec les poids, et `mlflow.pytorch` pour le Transformer écrit à la main, qui n'a pas de saveur dédiée. Chacun est enregistré au registre sous `syntra-<expérience>` et se recharge par ce nom.

Trois précisions sur ce dépôt de modèles :

- Les poids déposés sont ceux qui ont été **évalués**, relus depuis le meilleur checkpoint, et non l'objet en fin d'entraînement. Quand l'arrêt anticipé a retenu une époque antérieure, les deux diffèrent, et déposer le second stockerait des poids que personne n'a mesurés.
- Seul un run `OK` entre au registre. Un run `PARTIAL` a vu deux pas d'optimisation et huit documents de test : ses poids existent et ne veulent rien dire.
- Le modèle from scratch est sérialisé au format pickle et non au format graphe `pt2`, qui est le défaut de MLflow 3. Son `forward` prend une source, un masque et une cible décalée, et sa génération est une boucle Python : un graphe tracé refuserait l'appel ou en figerait une seule forme. `code_paths` copie `src` à côté des poids pour que l'objet se recharge sans que le dépôt soit installé.

Un échec de traçage n'interrompt jamais un run. Le résultat d'une expérience est l'enregistrement sur disque ; le magasin en est le miroir, et `python -m src.tracking.log --all` le reconstruit après coup.

## 8. Limites

Le corpus est un tirage de 20 000 exemples, pas CNN/DailyMail complet, qui en compte 287 113. Toutes les conclusions valent sur cette plage.

**La troncature à 512 tokens écarte la moitié du texte source et touche 85 % des articles.** C'est la limite dominante du projet sur ce corpus. Elle s'applique aux deux familles également, donc elle ne biaise pas la comparaison, mais elle abaisse le plafond atteignable par les deux.

Les deux familles ne diffèrent pas seulement par le pré-entraînement. `t5-small` compte 60,5 M paramètres contre 15,6 M pour le modèle from scratch, soit un facteur 3,9. Ce rapport ne sépare pas les deux causes : il mesure l'écart entre les deux dispositifs tels qu'ils sont, pas la part qui revient au pré-entraînement seul.

Chaque configuration n'est entraînée que sous une graine, 42. Les intervalles publiés sont des intervalles bootstrap sur les 1 000 documents de test : ils mesurent l'échantillonnage du jeu d'évaluation, pas la variance d'entraînement.

La profondeur est la seule grandeur d'architecture explorée. La largeur `d_model` et la taille du vocabulaire, que la section 2 désigne comme le levier plus probable, n'ont pas été balayées faute de budget de calcul.

Les limites qui se lisent sur les résultats seront écrites avec eux.

## Pour reproduire

```bash
make data                                     # corpus figé, sous graine 42
make reproduce MODE=full                      # les neuf expériences, les tableaux, les figures
```

Étape par étape :

```bash
python -m src.experiments.run --all           # les neuf expériences
make ablation                                 # les deux ablations
make figures                                  # les quatre figures
make report-sync                              # les tableaux de ce rapport
make corpus-sync                              # les tableaux du corpus
```

Les tableaux de ce rapport se relisent dans `reports/results/experiments.csv`. `make report-sync` et `make corpus-sync` les régénèrent depuis les enregistrements, dans `reports/_generated/` et directement entre les marqueurs de ce fichier.

Le magasin demande un `.env` rempli sur le modèle de `.env.example` :

```bash
make mlflow-ui                                # les runs et les modèles, port 5000
python -m src.tracking.log --all              # rejoue les enregistrements vers le magasin
```
