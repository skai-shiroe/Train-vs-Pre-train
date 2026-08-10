# Le Transformer expliqué simplement

Cette page raconte le même modèle que [Transformer from scratch](transformer.md), mais avec des images plutôt qu'avec des formules. La page technique reste la référence : celle-ci sert à comprendre *pourquoi* chaque pièce existe avant d'aller lire *comment* elle est écrite.

Aucun composant n'est inventé pour l'occasion. Chaque section renvoie au fichier de `src/models/scratch/` qu'elle décrit.

## L'idée en deux élèves

Imaginez deux élèves assis côte à côte.

**L'élève A, l'encodeur.** Il lit l'article en entier, du début à la fin, autant de fois qu'il le souhaite. Il n'écrit rien. Il annote seulement chaque mot avec ce que ce mot signifie *dans ce texte-là* : « chat » n'a pas le même voisinage dans un article animalier et dans un article sur un logiciel de messagerie.

**L'élève B, le décodeur.** Il rédige le résumé, un mot à la fois, et il n'a jamais le droit de lire l'article lui-même. À chaque mot qu'il veut écrire, il se tourne vers A et demande : « compte tenu de ce que j'ai déjà écrit, quelle partie de l'article dois-je regarder maintenant ? »

C'est exactement le schéma de `src/models/scratch/transformer.py` :

```text
source_ids ---> Encoder ---> memory
                              |
target_ids  ---> Decoder <----+ cross attention
                  |
                logits
```

La `memory` est le cahier d'annotations de l'élève A. La cross-attention est la question que B lui pose.

## L'exemple fil rouge

Document : **Le chat noir dort sur le canapé du salon.**

Résumé attendu : **Un chat dort.**

Toutes les sections qui suivent traitent ce même exemple.

## 1. Des mots vers des numéros

Le tokenizer T5 découpe le texte et rend des identifiants entiers. Le préfixe `summarize:`, espace finale comprise, est collé devant le document, dans `src/models/scratch/summarizer.py`.

```text
"summarize: Le chat noir dort sur le canapé du salon."
    -> [1783, 22, 4501, 913, 44, ..., 1]
```

Ce préfixe ne veut rien dire pour un modèle qui apprend son vocabulaire depuis zéro, mais le pipeline de données l'a mis devant chaque document d'entraînement. L'oublier à l'inférence reviendrait à présenter au modèle une distribution qu'il n'a jamais vue, et la baisse de score serait attribuée à l'architecture.

## 2. Des numéros vers des vecteurs

Un identifiant seul ne dit rien : `4501` n'est pas plus grand que `22`, il est simplement autre. On remplace donc chaque identifiant par un vecteur de 512 nombres, dans `src/models/scratch/embeddings.py`.

C'est une carte d'identité à 512 cases, apprise pendant l'entraînement. Rien ne dit à l'avance ce que contient chaque case ; à la fin de l'entraînement, les cartes de « chat » et de « chaton » se ressemblent, celles de « chat » et de « lundi » beaucoup moins.

Un détail compte : l'embedding est multiplié par la racine de `d_model`, soit environ 22,6 pour 512 dimensions. Sans ce facteur, les valeurs des cartes d'identité (de l'ordre de 0,04) seraient écrasées par le signal de position ajouté juste après, qui vit entre -1 et 1. On entendrait l'horloge et plus les mots.

## 3. L'horloge des positions

L'attention est indifférente à l'ordre. Sans précaution, « le chat mange la souris » et « la souris mange le chat » sont pour elle le même sac de mots. Le signal de position est donc ajouté explicitement, dans `src/models/scratch/positional_encoding.py`.

L'image utile est celle d'un tampon horodateur. Chaque position reçoit un motif d'ondes unique, produit par 256 cadrans qui tournent chacun à une vitesse différente. La position 3 et la position 4 ont des motifs proches, la position 3 et la position 90 non.

Deux propriétés rendent ce choix commode :

- il ne coûte aucun paramètre, puisque le motif est calculé et non appris ;
- pour un décalage fixe, le motif de `pos + k` s'obtient linéairement depuis celui de `pos`. Or une tête d'attention calcule précisément des combinaisons linéaires : la notion de « trois mots plus loin » lui est donc accessible.

## 4. L'attention, le coeur du mécanisme

Tout le modèle repose sur une seule formule, écrite dans `src/models/scratch/attention.py` :

```text
Attention(Q, K, V) = softmax(Q Kt / sqrt(d_k)) V
```

### Le trombinoscope

Chaque mot fabrique trois vecteurs à partir de sa carte d'identité :

| Vecteur | Nom | Ce qu'il représente |
| --- | --- | --- |
| **Q** | requête | « voici ce que je cherche », l'annonce que le mot passe |
| **K** | clé | « voici ce que j'offre », l'étiquette que le mot porte |
| **V** | valeur | le contenu réellement récupéré si ce mot est retenu |

Le mot **dort** lance sa requête : *qui est mon sujet ?* Sa requête est comparée aux clés de tous les mots de la phrase, ce qui donne un score par mot. Le softmax convertit ces scores en pourcentages dont la somme fait 100 %. Sur notre exemple, avec des valeurs illustratives :

```text
dort  ->  chat 0.61 | noir 0.14 | canapé 0.11 | salon 0.08 | le 0.06
```

On calcule alors la moyenne des valeurs, pondérée par ces pourcentages. Le vecteur de « dort » contient désormais une forte dose de « chat » : le mot a été contextualisé. Répété sur tous les mots en parallèle, c'est un simple produit de matrices.

### Pourquoi diviser par la racine de d_k

Sans cette division, plus les vecteurs sont larges, plus les scores s'étalent, et le softmax finit par répondre `[0, 0, 1, 0, 0]` au lieu d'une répartition nuancée.

Un softmax saturé a un gradient quasi nul : le modèle cesse d'apprendre. La division ramène la dispersion des scores à un niveau raisonnable et garde les comparaisons discutables, donc corrigibles.

### Pourquoi plusieurs têtes

Une tête d'attention ne suit qu'une relation à la fois. Or « dort » a besoin de savoir *qui* dort et *où*. Les 512 dimensions sont donc découpées en 8 têtes de 64 dimensions, dans `src/models/scratch/multi_head_attention.py`.

Au lieu d'un relecteur unique, on en met huit, chacun avec un surligneur de couleur différente : l'un traque les sujets, l'autre les compléments de lieu, un troisième les négations. La projection de sortie `W_O` recolle ensuite tous les surlignages en un seul avis.

Comme chaque tête est huit fois plus étroite, l'ensemble ne coûte pas plus cher qu'une seule tête large : l'expressivité est gagnée gratuitement.

## 5. Le bloc qui réfléchit tout seul

L'attention **déplace** de l'information entre les mots, mais seulement par moyennes pondérées, ce qui est une opération linéaire. Le réseau position par position de `src/models/scratch/feed_forward.py` fait l'inverse : il **transforme** chaque mot dans son coin, sans regarder ses voisins, avec une non-linéarité.

L'attention est la réunion d'équipe : on s'échange l'information. Le feed forward est le retour au bureau : chacun digère ce qu'il vient d'entendre. On étale les notes sur une grande table (`d_ff = 2048`) avant de les résumer sur une fiche (`d_model = 512`).

Alterner les deux six fois de suite, c'est ce qui donne sa valeur à la profondeur du modèle.

## 6. Les deux garde-fous de chaque couche

Dans `src/models/scratch/encoder_layer.py`, chaque sous-bloc est enveloppé de la même manière : une connexion résiduelle et une normalisation.

**Le résiduel, `x + f(x)`.** Aucune couche ne réécrit la copie à zéro : chacune ajoute des annotations en marge, et le texte de la couche précédente reste lisible dessous. C'est aussi une autoroute pour le gradient, qui atteint la première couche sans devoir traverser les six autres. Sans résiduels, une pile de six couches s'entraîne mal, et une pile plus profonde pas du tout.

**La normalisation de couche.** Elle remet le volume au même niveau après chaque sous-bloc, pour qu'une couche ne se mette pas à hurler pendant que les autres chuchotent.

Le projet normalise **avant** le sous-bloc et non après, ce qui laisse le chemin résiduel entièrement libre. C'est le réglage `norm_first: true`, qui s'entraîne de façon fiable sans le long échauffement du taux d'apprentissage que réclame la disposition de l'article original.

## 7. Les masques, les deux règles du jeu

C'est le point compris le plus tard, et il vit dans `src/models/scratch/masks.py`.

**Le masque de padding.** Dans un lot, toutes les séquences sont allongées au format de la plus longue avec du remplissage vide. Interdiction de recopier les lignes blanches de la feuille du voisin : sans ce masque, une prédiction dépendrait de la composition du lot, ce qui n'a aucun sens.

**Le masque causal.** Pendant l'entraînement, le résumé entier est donné au décodeur d'un seul coup. Sans précaution, au moment de prédire le troisième mot, le modèle voit le troisième mot. Le masque causal est le cache en carton qui descend sur la feuille : à la position 3, seules les positions 1, 2 et 3 sont visibles.

Sans lui, la perte s'effondre magnifiquement à l'entraînement, et la génération produit du charabia : le modèle a appris à recopier la réponse, pas à la deviner.

Les positions interdites reçoivent la plus petite valeur finie du type de calcul, et non moins l'infini. Sur une ligne entièrement masquée, moins l'infini donnerait une division de zéro par zéro, donc des NaN qui contamineraient tout le modèle.

## 8. L'entraînement, cinq exercices en un passage

La cible est décalée d'une position vers la droite, et la case libérée reçoit le token de départ. Le tokenizer T5 n'ayant pas de token de début de séquence, c'est l'identifiant de padding qui joue ce rôle, comme dans T5 lui-même.

| Position | Entrée du décodeur | Doit prédire |
| --- | --- | --- |
| 0 | `<start>` | `Un` |
| 1 | `<start> Un` | `chat` |
| 2 | `<start> Un chat` | `dort` |
| 3 | `<start> Un chat dort` | `.` |
| 4 | `<start> Un chat dort .` | `</s>` |

Les cinq exercices sont corrigés en un seul passage, en parallèle, parce que le masque causal garantit qu'aucune ligne ne triche sur la suivante. C'est ce qui rend un Transformer bien plus rapide à entraîner qu'un réseau récurrent, qui devrait dérouler les positions une par une.

La correction compare, à chaque position, les scores attribués aux 32 000 entrées du vocabulaire avec le bon mot, en ignorant les positions de remplissage.

Une économie au passage : la table qui transforme un identifiant en vecteur sert aussi, transposée, à retransformer un vecteur en scores de vocabulaire. C'est le même dictionnaire lu dans les deux sens. Le partage épargne environ 16 millions de paramètres sur 60, et force les vues d'entrée et de sortie d'un token à s'accorder.

## 9. La génération, quand le filet disparaît

À l'inférence il n'existe aucun résumé de référence. Le décodeur mange sa propre sortie, dans `src/models/scratch/generation.py` :

```text
<start>            -> le modèle produit "Un"
<start> Un         -> il produit "chat"
<start> Un chat    -> il produit "dort"
...                -> jusqu'au token de fin ou au budget de 64 tokens
```

L'encodeur ne tourne qu'une seule fois : le document ne change pas, seul le résumé s'allonge.

Trois façons de choisir le mot suivant sont implémentées :

- **Greedy search** prend à chaque pas le mot le plus probable. C'est le convive qui prend toujours le plat qui sent le meilleur maintenant, quitte à n'avoir plus faim pour le dessert. C'est aussi la référence honnête : tout gain annoncé ailleurs se mesure contre elle.
- **Beam search** garde en parallèle les `k` meilleurs débuts de résumé et ne tranche qu'à la fin. On remplit quatre assiettes et on choisit au moment de s'asseoir. Le score est divisé par la longueur élevée à une puissance, sans quoi les résumés courts gagneraient mécaniquement, ayant moins de termes négatifs à additionner.
- **Sampling** tire le mot au sort dans la distribution, après l'avoir resserrée par la température, `top_k` et `top_p`. Sans ce resserrement, la queue de plusieurs milliers de mots improbables pèse, cumulée, assez lourd pour sortir régulièrement : un résumé qui pioche dans la queue est un résumé qui invente. Cette stratégie ne sert jamais à mesurer un score, uniquement à une interface qui ne doit pas répondre deux fois la même phrase.

## Le parcours complet

```text
"Le chat noir dort sur le canapé du salon."
   |
   |  tokenizer : le texte devient des identifiants
   |  embedding x sqrt(512), puis horloge sinusoïdale
   v
+-- ENCODEUR, 6 couches ---------------------+
|  tous les mots se parlent, 8 têtes         |
|  chacun digère dans son coin               |
+--------------------------------------------+
   |
   |  memory : un vecteur par mot source,
   |  chacun conscient de tout le document
   v
+-- DÉCODEUR, 6 couches ---------------------+
|  1. le résumé se parle à lui-même, masqué  |  "qu'ai-je déjà écrit ?"
|  2. cross-attention vers la memory         |  "que dois-je lire ?"
|  3. chacun digère dans son coin            |
+--------------------------------------------+
   |
   |  projection vers 32 000 scores
   v
"Un chat dort."
```

## Ce qu'il faut retenir

La seule idée réellement nouvelle du Transformer tient en une phrase : **au lieu de faire circuler l'information de mot en mot comme dans une chaîne, chaque mot interroge tous les autres directement, en une opération matricielle parallélisable.**

Tout le reste est l'échafaudage qui rend cette idée entraînable :

| Pièce | Le problème qu'elle règle |
| --- | --- |
| Encodage positionnel | l'attention ne voit aucun ordre |
| Division par la racine de `d_k` | un softmax saturé n'a plus de gradient |
| Têtes multiples | une tête ne suit qu'une relation à la fois |
| Feed forward | l'attention ne fait que des moyennes, donc du linéaire |
| Connexions résiduelles | le gradient doit atteindre la première couche |
| Normalisation de couche | les échelles doivent rester comparables |
| Masque de padding | le remplissage n'est pas de l'information |
| Masque causal | prédire un mot qu'on a déjà lu n'apprend rien |

## Pour aller plus loin

- [Transformer from scratch](transformer.md) : la même architecture, avec les formules, les tests qui la garantissent et la référence de code.
- [Baseline pré-entraînée](baseline.md) : le T5 auquel ce modèle est comparé, sous les mêmes contraintes de décodage.
- [Entraînement](training.md) : ce qui se passe autour du modèle, de l'optimiseur aux points de reprise.
