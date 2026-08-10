# Baseline pré-entraînée

## Composants

```text
src/models/pretrained/
├── base.py      contrat commun : préfixe, troncature, génération, perte, persistance
├── t5.py        le modèle t5-small, et ce que T5 impose à la configuration
└── factory.py   résolution d'un nom d'expérience vers une classe
```

Le partage entre `base.py` et une sous-classe suit une règle simple : tout ce
qui ne dépend pas de l'architecture reste dans la base.

```text
base.py    préfixe et troncature, inférence par lots, adaptateur de perte,
           déplacement sur un device, sauvegarde, description d'un run
t5.py      quelle classe Hugging Face charger, et les règles que cette
           architecture impose à la configuration
```

## Ce que la baseline sert à mesurer

`t5-small` est le côté pré-entraîné de la comparaison, verrouillé par la section 2.1 du cahier des charges. Il est mesuré deux fois sur le même jeu de test.

```text
zero-shot     le modèle tel qu'il arrive du hub, sans aucun entraînement
fine-tuning   après entraînement sur chaque proportion du corpus de travail
```

L'écart entre les deux dit combien T5 doit à son pré-entraînement plutôt qu'aux 20 000 exemples que voit le modèle from scratch. C'est cet écart qui donne son sens à la courbe performance contre taille du corpus.

## Le préfixe n'est pas décoratif

T5 a été pré-entraîné sur un mélange de tâches supervisées, chacune sélectionnée par un préfixe textuel. Sans `"summarize: "` devant le document, espace final compris, le modèle n'est pas interrogé sur le résumé, et une mesure zero-shot prise ainsi rapporte un nombre qui ne dit rien de la tâche.

`T5Summarizer.check_config` refuse donc un préfixe vide, et le refus arrive avant le téléchargement des poids. Un run mal configuré échoue immédiatement au lieu de produire un score silencieusement faux.

Le préfixe et les longueurs de troncature ne sont pas saisis deux fois. `BaselineConfig.from_tokenizer_config` les dérive du bloc `tokenizer` de la configuration du corpus, ce qui garantit qu'à l'inférence un document est coupé exactement là où l'entraînement le coupait.

## Zero-shot et fine-tuning sont une seule classe

Rien ne change dans le code entre les deux : une baseline zero-shot est une baseline dont les poids n'ont jamais été modifiés. La distinction est portée par un drapeau explicite, jamais devinée.

```text
from_pretrained(config)                     mode zero_shot
from_pretrained(config, fine_tuned=True)    poids déjà entraînés, rechargés d'un dossier
from_checkpoint(config, payload["model"])   poids issus d'un checkpoint d'entraînement
```

`describe()` rapporte ce mode à côté de l'identifiant et du nombre de paramètres. Publier un score fine-tuné sous l'étiquette zero-shot serait un résultat inventé au sens de la section 44.

## Décodage partagé

Les deux modèles décodent à travers le même objet, `GenerationConfig`, remonté dans `src/models/generation.py` pour que ni le modèle from scratch ni la baseline n'en possède la définition.

```text
modèle from scratch   boucles de src/models/scratch/generation.py
baseline              méthode generate du modèle Hugging Face
configuration         src/models/generation.py, une seule
```

Ce n'est pas un rangement. Une largeur de faisceau ou une pénalité de longueur qui différerait entre les deux déplacerait le score rapporté pour une raison étrangère aux modèles. Un objet unique rend cette divergence inexprimable.

L'échantillonnage est coupé explicitement dans les arguments passés à `generate`. Un résumé échantillonné changerait d'une évaluation à l'autre, ce que la section 14 interdit pour un score rapporté.

Une dernière différence est corrigée à la sortie : un modèle encodeur décodeur Hugging Face renvoie le token de démarrage du décodeur en position zéro, le générateur from scratch non. Il est retiré ici, et les deux sorties restent directement comparables.

## Perte et boucle d'entraînement partagées

Il n'y a pas de seconde boucle d'entraînement. Le `Trainer` accepte une fonction de perte en paramètre, et `PretrainedSummarizer.batch_loss` fournit l'adaptateur.

```python
trainer = Trainer(
    summarizer.model,
    training_config,
    train_loader=train_loader,
    validation_loader=validation_loader,
    output_dir=run_dir,
    batch_loss=summarizer.batch_loss(training_config.label_smoothing),
    metadata=summarizer.describe(),
)
```

Le fine-tuning passe donc par la même moyenne pondérée par les tokens, le même écrêtage, les mêmes checkpoints et le même early stopping que le modèle from scratch. C'est ce qui rend les deux courbes d'entraînement comparables.

Le teacher forcing est laissé au modèle. Recevant `labels`, un modèle séquence à séquence Hugging Face les décale d'une position et démarre son décodeur sur le token de padding, exactement la convention du modèle from scratch. Les deux reçoivent la même entrée décodeur.

Quand le lissage de labels est nul, la perte calculée par le modèle est renvoyée telle quelle : la recalculer coûterait une entropie croisée inutile sur tout le vocabulaire à chaque pas. Au-dessus de zéro elle est recalculée ici, parce que le lissage est un hyperparamètre d'entraînement et non une propriété du modèle.

## Persistance

`save_pretrained` écrit les poids et le tokenizer au format Hugging Face, côte à côte.

```python
summarizer.save_pretrained(path)

reloaded = T5Summarizer.from_pretrained(
    BaselineConfig(hf_id=str(path), source_prefix="summarize: "),
    fine_tuned=True,
)
```

C'est le format que servira le registre de modèles de la section 20 et que chargera l'API de la section 27. Le tokenizer est écrit avec les poids pour que le dossier se recharge sur une machine qui n'a jamais vu le hub.

Lire un checkpoint d'entraînement reste à la charge de l'appelant, de sorte que `src/models/` ne dépende jamais de `src/training/` :

```python
payload = load_checkpoint(run_dir / "best.pt")
summarizer = T5Summarizer.from_checkpoint(config, payload["model"])
```

## Registre et extension

`factory.py` résout un nom d'expérience vers une classe, ce qui évite que la couche expérience importe un modèle directement.

Une seule baseline y est enregistrée. La section 9 cite mT5, mBART et MarianMT comme variantes possibles, et son arborescence mentionne `mbart.py` ; la section 2.1 verrouille la tâche au résumé en anglais et le modèle pré-entraîné à `t5-small`, ce qui ne laisse rien à faire à un modèle multilingue de traduction. Un wrapper qu'aucune expérience ne construit et qu'aucun test ne couvre serait du code mort qu'il faudrait quand même maintenir : il n'est pas écrit. Le registre est le point d'extension si cette décision est rouverte.

Un nom inconnu échoue avec un message qui liste les noms disponibles, plutôt qu'avec une `KeyError` nue.

## Ce que les tests garantissent

Les tests unitaires tournent sur un T5 de quelques milliers de paramètres et un tokenizer factice, donc hors ligne et en une fraction de seconde. Ils couvrent le câblage.

```text
un préfixe vide est refusé avant tout téléchargement
le préfixe est bien préposé à chaque document
la troncature respecte le budget configuré
la génération retire le token de démarrage du décodeur
les documents sont résumés par lots, pas en un seul tenseur
le lissage nul renvoie la perte du modèle sans la recalculer
la perte lissée ignore les positions de padding
la sauvegarde écrit les poids et le tokenizer
un checkpoint reconstruit une baseline marquée fine-tuned
```

Les tests d'intégration chargent les vrais poids de `t5-small`. Ils portent le marqueur `slow` et sortent du hook pre-push.

```text
la baseline et le corpus partagent la même instance de tokenizer
le checkpoint chargé pèse bien 60 millions de paramètres
la perte est inférieure à log(vocab_size), donc les poids portent une connaissance
le préfixe change réellement la sortie du modèle
le décodage est reproductible d'un appel à l'autre
le fine-tuning passe par le Trainer et fait baisser la perte de validation
une baseline sauvegardée se recharge sans le hub
```

Le troisième mérite une phrase. Un modèle qui n'a rien appris répartit sa masse de probabilité uniformément sur le vocabulaire, ce qui coûte `log(vocab_size)` par token. Une perte au-dessus de ce seuil signifierait que les poids ne sont jamais arrivés, ce qu'un test de forme ne verrait pas.

## Résultats mesurés

Les quatre expériences `t5-small` ont tourné le 8 août 2026, sur les 1 000 exemples du split test.

--8<-- "_generated/pretrained.md"

Le ROUGE-2 est plus parlant que le ROUGE-L sur ce que le fine-tuning apporte : il passe de 0,0304 en zero-shot à 0,0921 à 100 %, soit trois fois plus. La longueur générée le confirme, 36,4 mots en moyenne en zero-shot contre 21,3 pour la référence, et 17,3 après fine-tuning. Le modèle connaissait déjà la langue, il apprend le format.

Les exemples de code ci-dessus restent des appels et non des mesures. Les mesures sont dans `reports/results/`.

## Référence de code

::: src.models.pretrained.base

::: src.models.pretrained.factory
