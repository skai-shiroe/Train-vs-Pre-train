# Rapport

Entraîner un Transformer encodeur-décodeur pour le résumé automatique, puis le comparer à un modèle pré-entraîné de référence sur les mêmes données de test.

Ce rapport se lit seul : tous les chiffres qui portent une conclusion y figurent. Ils viennent de la campagne des 8 et 9 août 2026. Ses tableaux sont générés depuis les enregistrements de runs et injectés entre marqueurs, jamais saisis à la main.

| Exigence | Ce qui a été fait |
| --- | --- |
| Transformer encodeur-décodeur from scratch | Écrit composant par composant en PyTorch, 15,6 M paramètres, entraîné sur 10 %, 50 % et 100 % du corpus |
| Modèle pré-entraîné, zero-shot puis fine-tuné | `t5-small`, 60,5 M paramètres, une mesure zero-shot et trois fine-tunages |
| Ablation sur la taille du corpus | Sept runs, sous-ensembles emboîtés, budget d'époques constant |
| Métrique et analyse qualitative | ROUGE-1, ROUGE-2, ROUGE-L avec intervalles bootstrap à 95 %, plus une sélection d'exemples par run |

La campagne compte 9 expériences déclarées et 9 runs complets, aucun échec, pour 125 minutes de calcul cumulé sur une RTX 5060 Laptop sous `torch 2.13.0+cu130`. Chaque score est mesuré sur les mêmes 1 000 documents de test.

L'énoncé laisse le choix entre BLEU et ROUGE. La tâche étant du résumé, c'est ROUGE. BLEU est absent volontairement : aucune colonne vide ou remplie de zéros n'apparaît nulle part.

## 1. Le corpus

Le corpus de travail est un tirage figé de 22 000 exemples de **XSum** sous graine 42 : 20 000 pour l'entraînement, 1 000 pour la validation, 1 000 pour le test. Le « 100 % » des ablations désigne ces 20 000 exemples, jamais les 204 045 de XSum complet.

Le corpus est propre et étanche : aucun champ vide, aucun identifiant dupliqué, aucun document partagé entre les trois splits. Deux documents apparaissent deux fois dans l'entraînement, et un exemple porte un résumé plus long que son document. J'ai gardé les trois : nettoyer en silence le corpus de référence aurait rendu les scores incomparables avec la littérature XSum, ce qui coûte plus cher que trois exemples douteux sur 20 000.

Les sous-ensembles d'ablation sont emboîtés, 10 % préfixe de 50 %, lui-même préfixe de 100 %, vérifié par comparaison des identifiants. Tirés indépendamment, un écart entre deux points de la courbe mélangerait l'effet de la taille et celui de la composition de l'échantillon.

<!-- syntra:begin statistics -->
<!-- Généré par python -m src.data.fragments. Ne pas éditer à la main. -->

| Grandeur | Entraînement | Validation | Test |
| --- | --- | --- | --- |
| Mots par document, médiane | 296,5 | 287,5 | 302,0 |
| Mots par résumé, médiane | 21,0 | 21,0 | 21,0 |
| Tokens par document, moyenne | 525,3 | 533,2 | 535,2 |
| Tokens par document, médiane | 414,0 | 403,5 | 421,5 |
| Tokens par document, p95 | 1 305,0 | 1 329,8 | 1 380,4 |
| Tokens par résumé, médiane | 30,0 | 30,0 | 30,0 |
| Tokens par résumé, p95 | 43,0 | 43,0 | 42,0 |
| Taux de compression | 0,097 | 0,098 | 0,093 |
<!-- syntra:end statistics -->

La compression est le trait dominant du corpus. Le résumé médian fait 30 tokens pour un document médian de 414, soit 7,2 %. À ce niveau, recopier des phrases du source ne peut pas produire un bon score : la tâche est réellement abstractive. C'est aussi pourquoi un ROUGE-L de 0,23 sur XSum ne se compare pas à un ROUGE-L publié sur CNN/DailyMail, où la référence autorise la reprise de phrases entières.

Reste la troncature, qui est le compromis le plus coûteux du projet.

| Plafond source | Documents coupés | Texte conservé | Coût relatif de l'encodeur |
| --- | --- | --- | --- |
| 256 | 74,3 % | 44,0 % | 0,25x |
| **512** | **38,4 %** | **70,9 %** | **1,0x** |
| 768 | 20,3 % | 84,7 % | 2,2x |
| 1 024 | 10,8 % | 92,1 % | 4,0x |
| 1 536 | 2,5 % | 97,8 % | 9,0x |

À 512 tokens, 38 % des documents sont coupés et 29 % du texte source est perdu. Passer à 1 024 en récupérerait 21 points, pour quatre fois le coût de l'encodeur, l'attention étant quadratique. Le budget GPU disponible ne le permettait pas. Les deux familles de modèles subissent exactement la même troncature : ce qui est perdu, c'est une part du plafond atteignable, pas l'équité de la comparaison. Le plafond des cibles, 64 tokens, ne coupe que 0,42 % des résumés et est essentiellement gratuit.

## 2. Le Transformer from scratch

`torch.nn.Transformer` existe et fonctionne. L'objet du projet est de démontrer la compréhension de l'architecture, pas d'en consommer une implémentation. Chaque bloc est donc écrit séparément, et chaque décision de conception est couverte par un test qui échouerait si elle était fausse.

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

L'attention multi-têtes n'instancie pas `h` petites projections mais quatre projections larges, `W_Q`, `W_K`, `W_V` et `W_O`, toutes de `d_model` vers `d_model`, initialisées en Xavier uniforme avec des biais nuls. Le découpage en têtes est une opération de forme : le tenseur passe de `(2, seq, 256)` à `(2, 8, seq, 32)` par une vue et une transposition, l'attention s'applique tête par tête, puis le chemin inverse recompose `(2, seq, 256)` avant `W_O`. Les 32 dimensions par tête sont le `d_k` de la formule, et le coût total égale celui d'une tête unique de largeur `d_model`.

### Les masques

Le masque de padding empêche le modèle de lire le remplissage ; sans lui, les prédictions dépendraient de la composition du lot. Le masque causal empêche la position `t` de voir les positions suivantes pendant le teacher forcing ; sans lui, le modèle lit la réponse qu'on lui demande de prédire, la perte s'effondre et la génération reste aléatoire.

Un détail d'implémentation mérite d'être signalé : les positions interdites reçoivent la plus petite valeur finie du type, et non moins l'infini. Sur une séquence entièrement remplie de padding, moins l'infini donne une ligne de zéros divisée par zéro, donc des NaN. Un test couvre ce cas précis, parce qu'il ne se déclenche pas sur un lot ordinaire.

### Pre-norm plutôt que post-norm

```text
post-norm  x = LayerNorm(x + Sublayer(x))     article original
pre-norm   x = x + Sublayer(LayerNorm(x))     défaut du projet
```

Le pre-norm laisse le chemin résiduel libre de toute normalisation, donc le gradient atteint la première couche sans distorsion, et il s'entraîne sans le long warmup que le post-norm réclame. Pour un modèle entraîné from scratch sur un petit corpus, c'était le choix le moins risqué. Le post-norm reste accessible par `norm_first: false`, et les deux dispositions sont couvertes par les tests.

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

Une seule table sert l'encodeur, le décodeur et la projection de sortie. Elle est créée en premier et passée aux deux tours, pas recréée dans chacune : `encoder.embedding` et `decoder.embedding` sont le même objet en mémoire, et `decoder.output_projection.weight is embedding.weight` vaut `True`. L'attachement économise `vocab_size * d_model` paramètres, soit 15 591 424 au lieu de 23 809 024, et régularise un modèle entraîné sur peu de données en forçant les vues d'entrée et de sortie d'un token à s'accorder.

Ce tableau est le fait le plus important du modèle from scratch. La table d'embedding pèse plus que l'encodeur et le décodeur réunis, et le corpus ne permet pas de l'apprendre : sur les 32 100 entrées du tokenizer, 23 458 apparaissent au moins une fois, soit 73 %. Les 8 642 restantes représentent 2 212 352 paramètres, 14,2 % du modèle, qui ne reçoivent jamais le moindre gradient. La concentration aggrave le constat : 50 % des occurrences tiennent dans 100 types, et il faut 14 510 types pour couvrir 99 % du texte.

### Ce que les tests garantissent

Deux propriétés ne se voient pas sur une courbe de perte. Le décodeur ne lit pas le futur : modifier le token de l'entrée décodeur à la position `k` laisse les logits des positions antérieures strictement inchangés, et modifie ceux de la position `k` ; la seconde moitié de l'assertion compte autant que la première, sans quoi un modèle qui ignorerait entièrement son entrée passerait le test. Le padding ne change rien : ajouter du remplissage à la source laisse les logits identiques, et un masque mal diffusé fait échouer ce test alors que la perte continue de descendre.

S'y ajoutent la vérification de la formule d'attention sur une entrée construite à la main, l'absence de NaN sur une ligne entièrement masquée, la reproduction de l'encodage positionnel publié position par position, l'économie exacte de `vocab_size * d_model` paramètres par l'attachement, et le surapprentissage d'un lot unique jusqu'à moins de 20 % de la perte initiale. À l'initialisation, la perte vaut 10,9 sur un lot aléatoire contre 10,4 attendus pour une distribution uniforme sur 32 100 classes : c'est le contrôle le moins cher du projet, et une valeur très éloignée signale un câblage défectueux avant le premier entraînement.

## 3. La baseline pré-entraînée

`t5-small`, 60 506 624 paramètres, soit 3,9 fois le modèle from scratch. Elle est mesurée deux fois : en zero-shot, sans aucun entraînement, avec le préfixe de tâche `summarize:` ; puis fine-tunée sur 10 %, 50 % et 100 % du même corpus, avec le même budget d'époques et le même décodage que le modèle from scratch.

Les deux familles partagent le tokenizer, le corpus, le chargeur, la boucle d'entraînement, le décodage et la métrique. Le modèle zero-shot saute l'étape d'entraînement parce que ses poids ne bougent pas, pas parce qu'il emprunte un autre chemin.

Les quatre mesures sont prises sous la révision `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` de `t5-small`. Ce n'était pas le cas de la première série : les poids fine-tunés sont les nôtres, mais l'architecture dans laquelle ils sont chargés vient du hub, et sans révision fixée elle peut changer sans que rien ne change dans le dépôt. J'ai épinglé la révision et rejoué les quatre runs le 9 août plutôt que de garder des chiffres qu'un tiers n'aurait pas pu retrouver.

## 4. Protocole d'évaluation

Le bloc d'évaluation est identique dans les neuf fichiers d'expérience : `num_beams` 4, `max_new_tokens` 64, `no_repeat_ngram_size` 3, 1 000 rééchantillonnages bootstrap à 95 %. Quatre règles encadrent la mesure.

Une prédiction vide vaut zéro et reste dans la moyenne. Retirer les documents sur lesquels un modèle a échoué relèverait sa moyenne pour avoir échoué. Le compte des prédictions vides est reporté à côté du score, ce qui sépare un score faible d'un modèle cassé ; aucun des neuf modèles n'en a produit.

Chaque score porte un intervalle bootstrap à 95 %, calculé sous la graine du run. Sans lui, un écart de deux millièmes se lit comme un classement.

Seul un run complet remplit une colonne de score. Un run partiel garde sa ligne et son statut, ses cellules de score restent vides.

Des runs mesurés différemment ne sont pas mis dans un même tableau ni sur une même figure. Le budget de décodage, la largeur de faisceau et les réglages de ROUGE voyagent dans chaque enregistrement, et l'agrégation refuse d'écrire si deux runs d'une même étude divergent.

## 5. Résultats

| Expérience | ROUGE-1 | ROUGE-2 | ROUGE-L | IC 95 % sur ROUGE-L |
| --- | --- | --- | --- | --- |
| `pretrained_ft_100` | 0,2947 | 0,0898 | **0,2295** | [0,2227, 0,2365] |
| `pretrained_ft_50` | 0,2835 | 0,0808 | 0,2191 | [0,2129, 0,2256] |
| `pretrained_ft_10` | 0,2392 | 0,0585 | 0,1843 | [0,1782, 0,1900] |
| `scratch_100_layers2` | 0,2141 | 0,0442 | 0,1657 | [0,1605, 0,1710] |
| `scratch_100` | 0,2104 | 0,0439 | 0,1634 | [0,1581, 0,1686] |
| `scratch_100_layers6` | 0,2018 | 0,0395 | 0,1573 | [0,1525, 0,1627] |
| `scratch_50` | 0,2003 | 0,0385 | 0,1550 | [0,1499, 0,1600] |
| `pretrained_zero_shot` | 0,2028 | 0,0304 | 0,1366 | [0,1328, 0,1406] |
| `scratch_10` | 0,1573 | 0,0228 | 0,1250 | [0,1207, 0,1293] |

### Performance contre taille du corpus d'entraînement

C'est le livrable central du projet.

![Performance selon la taille du corpus](reports/figures/performance_vs_dataset_size.png)

<!-- syntra:begin families -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Proportion | Exemples | from scratch | `t5-small` fine-tuné | Écart absolu | Écart relatif |
| --- | --- | --- | --- | --- | --- |
| 10 % | 2 000 | 0,1250 | 0,1843 | 0,0593 | +47 % |
| 50 % | 10 000 | 0,1550 | 0,2191 | 0,0641 | +41 % |
| 100 % | 20 000 | 0,1634 | 0,2295 | 0,0662 | +40 % |
<!-- syntra:end families -->

Trois lectures, appuyées sur des intervalles disjoints à chaque point.

D'abord, l'écart ne se referme pas. L'écart relatif se resserre légèrement, de 47 % à 40 %, mais l'écart absolu grandit, de 0,0593 à 0,0662. Les deux courbes montent en parallèle, et donner plus de données au modèle from scratch ne le rapproche pas du pré-entraîné, qui en profite autant.

Ensuite, le pré-entraînement vaut plus que dix fois les données annotées. `pretrained_ft_10` atteint 0,1843 sur 2 000 exemples en 84 secondes d'entraînement, quand `scratch_100` atteint 0,1634 sur 20 000 exemples en 1 347 secondes : dix fois moins de données, seize fois moins de calcul, et un meilleur score.

Enfin, le rendement décroît des deux côtés. Pour le modèle from scratch, passer de 10 % à 50 % gagne 24 % en relatif, passer de 50 % à 100 % n'en gagne plus que 5 %.

### À partir de quelle taille le from-scratch devient-il compétitif ?

La réponse dépend de ce à quoi on le compare, et elle est différente dans les deux cas.

**Contre le pré-entraîné fine-tuné : jamais sur la plage mesurée.** Aux trois proportions, les intervalles de confiance des deux familles sont disjoints, et l'écart absolu grandit au lieu de se réduire. Il n'existe aucun point de croisement dans les données mesurées.

On peut chiffrer ce qu'il faudrait, à condition d'assumer les hypothèses. Le modèle from scratch gagne 0,0129 point de ROUGE-L par doublement du corpus entre 2 000 et 10 000 exemples, et seulement 0,0084 entre 10 000 et 20 000. Combler les 0,0662 qui le séparent de `pretrained_ft_100` demanderait :

| Hypothèse de pente | Doublements | Corpus nécessaire | Rapport à XSum complet |
| --- | --- | --- | --- |
| Optimiste, celle de 2 000 vers 10 000 | 5,1 | ~694 000 exemples | 3,4x |
| Récente, celle de 10 000 vers 20 000 | 7,9 | ~4 800 000 exemples | 23,5x |

Même l'hypothèse la plus favorable réclame plus de trois fois la totalité de XSum, qui compte 204 045 exemples. Et cette extrapolation est doublement optimiste : elle suppose une progression log-linéaire alors que le rendement observé décroît déjà, et elle suppose que le modèle pré-entraîné resterait immobile pendant que le from-scratch le rattrape, ce que les mesures démentent.

La raison de fond tient en un rapport. À 10 % du corpus, le modèle from scratch dispose de 743 100 tokens source. `t5-small` a été pré-entraîné sur de l'ordre de 34 milliards de tokens, environ 45 000 fois plus. La comparaison n'oppose pas deux modèles à données égales : elle oppose un modèle qui part de zéro à un modèle qui a déjà lu quatre ordres de grandeur de texte en plus.

**Contre le pré-entraîné zero-shot : entre 2 000 et 10 000 exemples.** C'est le seul seuil de compétitivité réellement observé, et il est net. À 2 000 exemples, le modèle from scratch obtient 0,1250 contre 0,1366 pour le zero-shot : il est derrière, et les intervalles [0,1207, 0,1293] et [0,1328, 0,1406] sont disjoints. À 10 000 exemples il obtient 0,1550, devant, et là encore les intervalles ne se recouvrent pas.

C'est une façon concrète de chiffrer ce que vaut le pré-entraînement sur cette tâche : environ le prix de quelques milliers d'exemples annotés, dès lors qu'on renonce à fine-tuner. Si on ne renonce pas, l'entraînement from scratch coûte plus cher et rend moins, à toutes les tailles mesurées.

### Ablation d'architecture

Une seconde ablation fait varier la profondeur, tout le reste étant identique, corpus compris.

<!-- syntra:begin architecture -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Expérience | Couches | Paramètres | ROUGE-L | IC 95 % | Entraînement |
| --- | --- | --- | --- | --- | --- |
| `scratch_100_layers2` | 2 + 2 | 11 905 024 | 0,1657 | [0,1605, 0,1710] | 838 s |
| `scratch_100` | 4 + 4 | 15 591 424 | 0,1634 | [0,1581, 0,1686] | 1 347 s |
| `scratch_100_layers6` | 6 + 6 | 19 277 824 | 0,1573 | [0,1525, 0,1627] | 1 882 s |
<!-- syntra:end architecture -->

Le résultat est négatif et il est publié tel quel. Le ROUGE-L décroît quand la profondeur augmente. Aucune paire n'est séparée au seuil de 95 %, tous les intervalles se recouvrent, donc aucune différence individuelle n'est établie. Ce qui reste, c'est que trois runs indépendants classent dans le même sens, et qu'aucun gain n'apparaît là où la profondeur coûte 2,2 fois plus de calcul.

La perte de validation classe d'ailleurs différemment : elle place la profondeur 4 en tête et sature ensuite, quand le ROUGE place la profondeur 2 en tête. Ce n'est pas contradictoire. La perte mesure la prédiction du token suivant sous forçage par la référence, le ROUGE mesure un texte produit en génération autorégressive avec faisceau ; un modèle peut gagner sur la première sans gagner sur le second. C'est la justification concrète du choix de reporter le ROUGE.

Le compte de paramètres explique le non-résultat. La table d'embedding représente 69 % du modèle à 2 couches et 53 % du modèle à 4 couches, et 14 % du modèle entier n'est jamais mis à jour faute d'occurrences. Faire varier la profondeur ne fait donc varier qu'une minorité des paramètres, pendant que la majorité reste sous-entraînée. À `d_model` 256 et avec le vocabulaire T5, le levier n'est pas la profondeur.

## 6. Analyse qualitative

Chaque run sélectionne son meilleur cas, son pire cas et un tirage aléatoire, de façon déterministe. Le tirage compte autant que les extrêmes, qui sont par construction les deux exemples les moins représentatifs. Un même document de test, résumé par les trois modèles :

> **Référence.** Three men have been arrested on suspicion of murder over the shooting of a man at a meat market.
>
> **`pretrained_ft_100`**, ROUGE-L 0,714. Three men have been arrested on suspicion of murder after a man was shot in the head and chest at a meat market.
>
> **`scratch_100`**, ROUGE-L 0,541. people have been arrested on suspicion of murder after a man was stabbed to death in a crash.
>
> **`pretrained_zero_shot`**, ROUGE-L 0,123. the 44-year-old was found badly injured at the Stanley Meat Market in the old Swan area of Liverpool on 27 January. he was shot in the head and chest and died later in hospital. the three men arrested remain in police custody for questioning.

Le zero-shot ne résume pas, il recopie : trois phrases extraites du document, là où la référence en demande une. Ses sorties font en moyenne 36,4 mots contre 21,3 pour la référence, quand tous les modèles entraînés se calent entre 17,0 et 19,4 mots. C'est la mesure de ce que le fine-tuning apprend ici : le format XSum plutôt que la langue, que `t5-small` connaît déjà.

Le modèle from scratch, lui, a appris la forme et invente le fond. La structure de phrase est correcte et idiomatique, mais l'homme a été poignardé au lieu d'être abattu, et le lieu est devenu un accident de la route. C'est le comportement attendu d'un modèle qui a vu 20 000 exemples : il apprend à quoi ressemble un résumé XSum bien avant d'apprendre à lire le document. Un défaut mesurable accompagne cela, visible ci-dessus avec « people » pour « Three people » : le modèle from scratch commence fréquemment son résumé par un fragment de sous-mot au lieu d'un mot capitalisé.

<!-- syntra:begin capitalisation -->
<!-- Généré par python -m src.experiments.fragments. Ne pas éditer à la main. -->

| Modèle | Prédictions commençant par une minuscule |
| --- | --- |
| `pretrained_ft_10` | 2,0 % |
| `pretrained_ft_50` | 0,0 % |
| `pretrained_ft_100` | 0,1 % |
| `scratch_10` | 100,0 % |
| `scratch_50` | 99,6 % |
| `scratch_100` | 62,9 % |
| `scratch_100_layers2` | 73,7 % |
| `scratch_100_layers6` | 84,7 % |
| `pretrained_zero_shot` | 87,7 % |
| Références | 0,0 % |
<!-- syntra:end capitalisation -->

La décroissance est régulière avec la taille du corpus, ce qui montre un apprentissage incomplet et non un défaut de code : la génération partage le même découpage de séquence pour les deux familles. Le zero-shot est haut pour une autre raison, visible dans l'exemple ci-dessus : il recopie des phrases prises au milieu du document, donc au milieu d'une phrase. La majuscule initiale est la convention typographique la plus élémentaire du corpus, et le modèle from scratch ne l'apprend qu'entre 10 000 et 20 000 exemples, encore imparfaitement.

## 7. Limites

Le corpus est un tirage de 20 000 exemples, pas XSum complet. Toutes les conclusions valent sur cette plage, et l'extrapolation de la section 5 est présentée comme une extrapolation, avec ses hypothèses.

La troncature à 512 tokens écarte 29 % du texte source. Elle s'applique aux deux familles également, donc elle ne biaise pas la comparaison, mais elle abaisse le plafond atteignable par les deux.

Les deux familles ne diffèrent pas seulement par le pré-entraînement. `t5-small` compte 60,5 M paramètres contre 15,6 M pour le modèle from scratch, soit un facteur 3,9. La comparaison oppose donc un modèle pré-entraîné et large à un modèle initialisé au hasard et plus petit, et ce rapport ne sépare pas les deux causes : tout ce qu'il mesure, c'est l'écart entre les deux dispositifs tels qu'ils sont, pas la part qui revient au pré-entraînement seul. Un modèle from scratch porté à 60 M paramètres trancherait, et la section 5 donne la raison de ne pas l'avoir tenté : la profondeur n'achète rien ici, et le levier serait `d_model` et le vocabulaire, hors du budget disponible.

Chaque configuration n'a été entraînée que sous une graine, 42. Les intervalles publiés sont des intervalles bootstrap sur les 1 000 documents de test : ils mesurent l'échantillonnage du jeu d'évaluation, pas la variance d'entraînement. Les conclusions qui reposent sur des écarts larges, la supériorité du pré-entraîné à chaque proportion et le seuil de compétitivité contre le zero-shot, ne dépendent pas de ce point. Deux lectures plus fines en dépendent : la croissance de l'écart absolu, qui gagne 0,0069 entre 10 % et 100 % sans qu'aucun test ne porte sur cette différence, et le classement de l'ablation d'architecture, dont les trois intervalles se recouvrent. Les rejouer sous trois graines et publier moyenne et écart-type est ce qu'il faudrait faire avant de leur donner plus de poids.

La profondeur est la seule grandeur d'architecture explorée. La largeur `d_model` et la taille du vocabulaire, que la section 2 désigne comme le levier plus probable, n'ont pas été balayées faute de budget de calcul. C'est ce que je reprendrais en premier avec une machine plus grosse.

**Aucun des neuf runs n'est rattachable à un commit.** Les cinq runs `scratch_*` portent `git_commit: unknown`. Les quatre runs `pretrained_*` portent `5a9c95df844f`, et ce commit n'existe nulle part : ni dans ce dépôt, ni sur `origin`, qui ne porte que `main` et `develop`. La campagne a tourné dans un répertoire de travail distinct, `Scolaire/Syntra`, qui n'a pas de `.git` ; le dépôt courant est un clone du 10 août 2026 et les commits locaux de cet arbre n'ont jamais été poussés. Ils sont perdus définitivement. Les neuf runs portent par ailleurs `git_dirty: true`, donc même retrouvé, ce commit aurait situé la campagne sans la reconstituer.

Le code, lui, n'était pas perdu : il était encore sur le disque, sans rien pour le protéger. Il est désormais figé, avec les mesures, sur la branche orpheline `archive/campagne-2026-08` — `reports/results` est gitignoré et n'existait donc qu'en deux copies non sauvegardées. Ce qui remplace le commit manquant est une comparaison plutôt qu'une affirmation, et la différence compte : un hash se croit, une comparaison se rejoue.

```bash
python -m scripts.compare_archive     # src d'aujourd'hui contre l'arbre de campagne
```

Les deux arbres `src` sont comparés après suppression des commentaires et des docstrings, sur les arbres syntaxiques. Au 13 août 2026 : **65 fichiers sur 71 sont logiquement identiques**, dont l'attention, le Transformer, la baseline pré-entraînée, l'entraîneur, la sélection du matériel et le tracking. Cinq fichiers ont changé et un seul existait dans la campagne. Quatre des cinq — `data/fragments.py`, `experiments/fragments.py`, `experiments/reproduce.py`, `utils/markdown.py` — sont en aval de la mesure : ils génèrent les tableaux de ce rapport et le rendent en Markdown. Le cinquième, `experiments/run.py`, a reçu depuis un avertissement qui s'imprime sur la sortie d'erreur avant la première expérience quand le commit est inconnu ou l'arbre sale : ce que la campagne aurait dû lire avant de commencer plutôt que de le découvrir dans ses propres enregistrements. `experiments/publish.py`, présent seulement dans l'archive, publiait le site MkDocs supprimé depuis. Aucun des six n'entre dans le calcul d'un chiffre de la section 5.

Cela ne remplace pas un commit et ne prétend pas le faire. Un lecteur qui veut vérifier que le dépôt produit encore ces chiffres doit relancer la campagne ; ce que la comparaison établit, c'est que le chemin de calcul n'a pas bougé entre-temps.

Enfin, les tableaux de ce rapport sont générés depuis les enregistrements de runs et injectés entre marqueurs par `make report-sync` et `make corpus-sync` : une nouvelle campagne les réécrit, et un tableau décrivant la campagne précédente n'est plus un état atteignable. Les quantités citées à l'intérieur des phrases, elles, restent écrites à la main et revérifiées à l'œil contre `reports/results/experiments.csv`, la dernière fois le 10 août 2026. C'est là que se logerait désormais une divergence.

## Pour reproduire

```bash
make data                                     # corpus figé, sous graine 42
python -m src.experiments.run --all           # les neuf expériences
make ablation                                 # les tableaux
make figures                                  # les quatre figures
```

Les tableaux de ce rapport se relisent dans `reports/results/experiments.csv`. `make report-sync` et `make corpus-sync` les régénèrent depuis les enregistrements, dans `reports/_generated/` et directement entre les marqueurs de ce fichier.

La campagne d'origine ne se rejoue pas : ses enregistrements et le code qui les a produits sont figés sur `archive/campagne-2026-08`, pour la raison exposée en section 7. `git show archive/campagne-2026-08:ARCHIVE.md` en donne le détail, et `python -m scripts.compare_archive` mesure ce qui a bougé depuis. Le magasin MLflow, lui, se reconstruit depuis les enregistrements, puisqu'il n'en est que le miroir :

```bash
python -m src.tracking.log --all --tracking-uri sqlite:///mlflow.db
make mlflow-ui                                # les neuf runs, port 5000
```
