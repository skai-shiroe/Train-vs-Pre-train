<#
.SYNOPSIS
    Les cibles du Makefile, appelables depuis PowerShell.

.DESCRIPTION
    Le Makefile reste la reference : il tourne en CI et sous Git Bash. Ce script
    en est la porte d'entree Windows, pour ne pas avoir a ouvrir Git Bash juste
    pour lancer une cible. Les deux fichiers listent les memes commandes, donc
    une cible ajoutee d'un cote doit l'etre de l'autre.

.EXAMPLE
    .\make.ps1                      # la liste des cibles
    .\make.ps1 reproduce -Mode full # la campagne complete
    .\make.ps1 test
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $Target = 'help',

    # Mode de la cible reproduce. quick verifie la chaine, full produit la campagne.
    [ValidateSet('quick', 'full')]
    [string] $Mode = 'quick',

    # cu130 et non cu128 : le plancher torch est a 2.13 pour GHSA-rrmf-rvhw-rf47,
    # et l'index cu128 s'arrete a 2.11. Demande un pilote compatible CUDA 13.
    [string] $TorchIndex = 'https://download.pytorch.org/whl/cu130'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Repli sur l'interpreteur du PATH quand le venv du depot est absent ou
# incomplet. Un clone frais n'en a pas, et l'erreur sur un fichier introuvable
# ne dirait rien du vrai probleme.
#
# Mais le repli est verifie, contrairement a celui du Makefile : sur cette
# machine 'python' est un 3.14, hors de la plage que pyproject.toml declare.
# Sans ce controle, 'install' deverserait torch dans le Python systeme et
# 'test' echouerait sur un ModuleNotFoundError qui ne dit pas pourquoi.
#
# Resolu a la demande et non au chargement : 'help' et 'clean' n'appellent
# aucun interpreteur et doivent repondre sur un depot sans venv.
$script:PY = $null

function Resolve-Python {
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) { return $venvPython }

    $version = (& 'python' -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null)
    Write-Warning ".venv absent : repli sur le 'python' du PATH (version $version)."
    if ($version -and [version]$version -ge [version]'3.14') {
        throw @"
Python $version est hors de la plage du depot (>=3.12,<3.14).
Recreer le venv avant toute autre cible :
    py -3.12 -m venv .venv
    .\make.ps1 install
"@
    }
    return 'python'
}

$CovArgs = @(
    '--cov=src'
    '--cov-report=xml:reports/coverage.xml'
    '--cov-report=term-missing'
    '--cov-fail-under=80'
)

# Chaque etape s'arrete sur un code de retour non nul, comme make. Sans ca une
# cible enchainee continuerait sur l'echec de la precedente.
function Invoke-Step {
    param([string] $Exe, [string[]] $Arguments)
    Write-Host "  $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Echec ($LASTEXITCODE) : $Exe $($Arguments -join ' ')"
    }
}

function Invoke-Python {
    param([string[]] $Arguments)
    if (-not $script:PY) { $script:PY = Resolve-Python }
    Invoke-Step -Exe $script:PY -Arguments $Arguments
}

# Le magasin vient de .env, jamais d'un fichier suivi : une URI PostgreSQL porte
# un mot de passe.
function Read-DotEnv {
    $path = Join-Path $PSScriptRoot '.env'
    $values = @{}
    if (-not (Test-Path $path)) { return $values }
    foreach ($line in Get-Content $path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
        $key, $value = $trimmed -split '=', 2
        $values[$key.Trim()] = $value.Trim().Trim('"', "'")
    }
    return $values
}

$Targets = [ordered]@{
    'help'             = 'Affiche les cibles disponibles'
    'install'          = 'Installe les dependances'
    'kernel'           = 'Enregistre le kernel Jupyter du depot, demande le groupe eda'
    'test'             = 'Lance toute la suite'
    'test-unit'        = 'Lance les tests unitaires'
    'test-integration' = "Lance les tests d'integration"
    'coverage'         = 'Lance les tests avec le seuil de couverture de 80 pour cent'
    'data'             = 'Telecharge, valide et prepare le corpus de travail'
    'train-scratch'    = 'Entraine le Transformer from scratch sur 100 pour cent du corpus'
    'train-pretrained' = 'Fine-tune T5 sur 100 pour cent du corpus'
    'evaluate'         = 'Evalue la baseline zero-shot sur le jeu de test commun'
    'ablation'         = 'Rejoue les ablations taille de corpus et architecture'
    'figures'          = 'Trace les quatre figures dans reports/figures'
    'mlflow-ui'        = "Sert l'interface MLflow sur le magasin configure, port 5000"
    'report-sync'      = 'Regenere les tableaux de resultats a partir des enregistrements de runs'
    'corpus-sync'      = 'Regenere les tableaux du corpus a partir du manifeste et des statistiques'
    'reproduce'        = 'Reproduit la chaine scientifique complete. -Mode full ou quick'
    'clean'            = 'Supprime les caches et les rapports generes'
}

switch ($Target) {

    'help' {
        Write-Host 'Cibles disponibles :'
        foreach ($name in $Targets.Keys) {
            Write-Host ('  {0,-18} {1}' -f $name, $Targets[$name])
        }
        Write-Host ''
        Write-Host 'Exemple : .\make.ps1 reproduce -Mode full'
    }

    'install' {
        Invoke-Python @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Python @('-m', 'pip', 'install', 'torch>=2.13,<3.0', '--index-url', $TorchIndex)
        Invoke-Python @('-m', 'pip', 'install', '-e', '.[dev]')
    }

    # Les notebooks declarent le kernel 'train-vs-pre-train'. Sans cette cible,
    # Jupyter leur donne le 'python3' qu'il trouve, dont l'argv est un 'python'
    # nu resolu depuis le PATH : ici un 3.14 sans torch ni syntra.
    # --sys-prefix ecrit le kernel dans .venv, donc il suit l'environnement.
    #
    # Le nom porte celui du depot et non celui du paquet : un kernel appele
    # 'syntra' se confond avec le projet voisin du meme nom, et la confusion
    # coute une session entiere a chaque fois.
    'kernel' {
        Invoke-Python @('-m', 'pip', 'install', '-e', '.[eda]')
        Invoke-Python @('-m', 'ipykernel', 'install', '--sys-prefix',
                        '--name', 'train-vs-pre-train', '--display-name', 'Train-vs-Pre-train (.venv)')
    }

    'test'             { Invoke-Python @('-m', 'pytest') }
    'test-unit'        { Invoke-Python @('-m', 'pytest', 'tests/unit') }
    'test-integration' { Invoke-Python @('-m', 'pytest', 'tests/integration') }

    'coverage' {
        New-Item -ItemType Directory -Force -Path 'reports' | Out-Null
        Invoke-Python (@('-m', 'pytest') + $CovArgs + @('--junitxml=reports/junit.xml'))
    }

    'data' {
        Invoke-Python @('-m', 'src.data.build', '--config', 'configs/data/cnn_dailymail.yaml')
    }

    'train-scratch' {
        Invoke-Python @('-m', 'src.experiments.run', '--config', 'configs/experiments/scratch_100.yaml')
    }

    'train-pretrained' {
        Invoke-Python @('-m', 'src.experiments.run', '--config', 'configs/experiments/pretrained_ft_100.yaml')
    }

    'evaluate' {
        Invoke-Python @('-m', 'src.experiments.run', '--config', 'configs/experiments/pretrained_zero_shot.yaml')
    }

    'ablation' {
        Invoke-Python @('-m', 'src.experiments.ablation', '--study', 'dataset_size')
        Invoke-Python @('-m', 'src.experiments.ablation', '--study', 'architecture')
    }

    # Les figures lisent les memes enregistrements que les tableaux, pas les CSV
    # qu'ils produisent : une figure tracee depuis une valeur arrondie ne dit
    # plus la meme chose que le tableau a cote.
    'figures' { Invoke-Python @('-m', 'src.experiments.figures') }

    # Sans .env la cible s'arrete en le disant, plutot que de servir un magasin
    # local vide qui ressemble a une campagne perdue.
    'mlflow-ui' {
        $dotenv = Read-DotEnv
        $uri = if ($dotenv.ContainsKey('MLFLOW_TRACKING_URI')) { $dotenv['MLFLOW_TRACKING_URI'] } else { $null }
        if (-not $uri) {
            Write-Error 'MLFLOW_TRACKING_URI absent. Copier .env.example en .env et le remplir.'
            exit 1
        }
        $root = if ($dotenv.ContainsKey('MLFLOW_ARTIFACT_ROOT') -and $dotenv['MLFLOW_ARTIFACT_ROOT']) {
            $dotenv['MLFLOW_ARTIFACT_ROOT']
        } else {
            'mlartifacts'
        }
        Invoke-Python @('-m', 'mlflow', 'ui', '--backend-store-uri', $uri, '--default-artifact-root', $root)
    }

    # Cette cible lit reports/results, que .gitignore garde hors du depot. Sur
    # une machine sans campagne elle reecrirait tous les tableaux en NOT_RUN,
    # ce qui est la raison pour laquelle elle reste manuelle.
    'report-sync' { Invoke-Python @('-m', 'src.experiments.fragments') }

    # Separee de report-sync, avec une autre source : celle-ci lit data/processed.
    # Une machine peut porter un corpus sans campagne, et l'inverse.
    'corpus-sync' { Invoke-Python @('-m', 'src.data.fragments') }

    # -u pour que les lignes de syntra.training s'affichent au fil de l'eau
    # plutot que par blocs : une campagne est longue, on la suit en direct.
    'reproduce' { Invoke-Python @('-u', '-m', 'src.experiments.reproduce', '--mode', $Mode) }

    'clean' {
        foreach ($path in '.pytest_cache', 'htmlcov', '.coverage',
                          'reports/junit.xml', 'reports/coverage.xml') {
            if (Test-Path $path) {
                Remove-Item -Recurse -Force $path
                Write-Host "  supprime $path" -ForegroundColor DarkGray
            }
        }
    }

    default {
        Write-Error "Cible inconnue : '$Target'. Lancer .\make.ps1 pour la liste."
        exit 1
    }
}
