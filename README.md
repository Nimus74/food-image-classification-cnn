# GourmetAI — Classificazione di immagini di cibo

Un progetto didattico di **Computer Vision** basato su **Transfer Learning** e **PyTorch**.  
L'obiettivo è classificare automaticamente immagini di piatti in 14 categorie alimentari, confrontando più architetture di reti neurali per selezionare quella con le migliori prestazioni.

Documentazione del notebook `gourmetai_food_classifier_v1.ipynb`, versione definitiva del progetto con supporto a **checkpoint progressivi** (fault tolerance).

---

## Indice

1. [Descrizione del progetto](#1-descrizione-del-progetto)
2. [Dataset](#2-dataset)
3. [Struttura del progetto](#3-struttura-del-progetto)
4. [Checkpoint e fault tolerance](#4-checkpoint-e-fault-tolerance)
5. [Flusso operativo](#5-flusso-operativo)
6. [Architetture testate](#6-architetture-testate)
7. [Tecnologie e dipendenze](#7-tecnologie-e-dipendenze)
8. [Come eseguire il notebook](#8-come-eseguire-il-notebook)
9. [Output prodotti](#9-output-prodotti)
10. [Suggerimenti per migliorare la precisione](#10-suggerimenti-per-migliorare-la-precisione)

---

## 1. Descrizione del progetto

GourmetAI è una pipeline completa di classificazione di immagini food che:

- Addestra una **CNN baseline** da zero come punto di riferimento
- Confronta **6 modelli di Transfer Learning** (da architetture classiche a Transformer)
- Seleziona automaticamente il **modello migliore** in base alla validation accuracy
- Produce metriche di valutazione dettagliate (confusion matrix, classification report) sul test set
- Genera automaticamente un **report HTML** con tutti i risultati al termine di ogni esecuzione
- Include **progressive checkpointing**: ogni N epoch viene salvato un checkpoint completo, consentendo di riprendere il training dopo crash, interruzione del kernel o timeout di Google Colab

Il progetto è strutturato con finalità didattiche: la progressione storica dei modelli (VGG16 → ResNet → EfficientNet → ConvNeXt → Swin Transformer) consente di osservare direttamente il miglioramento delle prestazioni ottenibile adottando architetture più moderne.

### Funzionalità principali (checkpoint)

| Funzionalità                       | Descrizione                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| Best model (EarlyStopping)         | Salvataggio del miglior modello per validation loss                         |
| Checkpoint periodico               | `resume_head.pt`, `resume_finetune.pt` ogni `checkpoint_every` epoch        |
| Ripristino automatico              | Rilevamento e caricamento automatico dal checkpoint in caso di interruzione |
| Stato EarlyStopping serializzabile | `state_dict` / `load_state_dict` per ripristino coerente                    |
| Parametro `checkpoint_every`       | In Config (default = 5), disabilitabile con 0                               |

---

## 2. Dataset

| Split       | Immagini   |
|-------------|------------|
| Train       | 8 960      |
| Validation  | 2 240      |
| Test        | 2 800      |
|-------------|------------|
| **Totale**  | **14 000** |

**14 classi alimentari:**

|--------------|----------------|---------------|-----------|
| Baked Potato | Crispy Chicken | Donut         | Fries     |
| Hot Dog      | Sandwich       | Taco          | Taquito   |
| Apple Pie    | Cheesecake     | Chicken Curry | Ice Cream |
| Omelette     | Sushi          |               |           |

Il dataset è disponibile pubblicamente:

```
https://proai-datasets.s3.eu-west-3.amazonaws.com/dataset_food_classification.zip
```

La cella di setup scarica e decomprime automaticamente il dataset se la cartella `dataset/` non è presente. Struttura attesa (ImageFolder standard):

```
dataset/
├── train/
│   ├── apple_pie/
│   ├── cheesecake/
│   └── ...
├── val/
└── test/
```

---

## 3. Struttura del progetto

```
Project/
├── gourmetai_food_classifier_v1.ipynb  ← notebook principale (questo progetto)
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/
│   ├── baseline/
│   │   └── best_baseline.pt
│   ├── vgg16/
│   │   ├── best_head.pt
│   │   ├── best_finetune.pt
│   │   ├── resume_head.pt
│   │   └── resume_finetune.pt
│   ├── resnet50/
│   ├── efficientnet_b0/
│   ├── efficientnet_b4/
│   ├── convnext_tiny/
│   └── swin_tiny/
└── reports/
    └── report_YYYY-MM-DD_HH-MM-SS.html
```

---

## 4. Checkpoint e fault tolerance

### Struttura dei checkpoint

Per ogni modello TL addestrato, la cartella `checkpoints/{model_name}/` conterrà fino a 4 file:

| File                 | Descrizione                                             |
|----------------------|---------------------------------------------------------|
| `best_head.pt`       | Pesi del miglior modello (fase head-only)               |
| `best_finetune.pt`   | Pesi del miglior modello (fase fine-tuning)             |
| `resume_head.pt`     | Checkpoint periodico fase head (fault tolerance)        |
| `resume_finetune.pt` | Checkpoint periodico fase fine-tuning (fault tolerance) |

I file `resume_*.pt` vengono sovrascritti ad ogni checkpoint periodico e **non** sono il best model: contengono tutto il necessario per riprendere il training.

### Struttura del checkpoint periodico

```python
{
    "epoch": <int>,                      # ultima epoch completata
    "model_state": <state_dict>,         # pesi del modello all'epoch salvata
    "optimizer_state": <state_dict>,     # stato ottimizzatore (momentum, lr)
    "scheduler_state": <state_dict>,     # stato scheduler lr (None se assente)
    "early_stopping_state": {            # stato EarlyStopping
        "min_val_loss": <float|None>,
        "counter": <int>,
        "early_stop": <bool>
    },
    "history": {                         # metriche accumulate fino all'epoch salvata
        "train_loss": [...],
        "val_loss": [...],
        "val_acc": [...]
    }
}
```

### Come funziona il ripristino

1. Il training viene interrotto per qualsiasi motivo (crash, timeout, stop manuale).
2. Si riesegue la stessa cella del loop TL.
3. La funzione `train()` controlla se il file `resume_head.pt` (o `resume_finetune.pt`) esiste.
4. Se esiste, carica automaticamente: pesi del modello, stato ottimizzatore, stato scheduler, stato EarlyStopping, history parziale.
5. Il training riprende dall'epoch successiva a quella salvata.
6. Se il file non esiste, il training parte da zero normalmente.

> Non è necessario modificare nulla nel codice per riprendere: il rilevamento è automatico.

Per azzerare e ripartire da zero, eliminare i file `resume_head.pt` e `resume_finetune.pt` dalla cartella del modello:

```bash
rm checkpoints/resnet50/resume_head.pt
rm checkpoints/resnet50/resume_finetune.pt
```

---

## 5. Flusso operativo

Il notebook `gourmetai_food_classifier_v1.ipynb` esegue le seguenti fasi in sequenza:

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE GOURMETAI                       │
├──────────────┬──────────────────────────────────────────────────┤
│  SETUP       │ Import librerie, seed, device (CPU/MPS/CUDA)     │
├──────────────┼──────────────────────────────────────────────────┤
│  DATI        │ Transforms (baseline + augmentation ImageNet)    │
│              │ Dataset (ImageFolder) + DataLoader               │
│              │ Visualizzazione campioni e augmentation          │
├──────────────┼──────────────────────────────────────────────────┤
│  BASELINE    │ Addestramento CNN semplice (da zero)             │
│              │ Valutazione su val e test → benchmark            │
├──────────────┼──────────────────────────────────────────────────┤
│  TRANSFER    │ Loop su 6 modelli candidati:                     │
│  LEARNING    │   1. Head-only (backbone congelato)              │
│              │   2. Fine-tuning (ultimi blocchi sbloccati)      │
│              │   → salvataggio checkpoint (best + resume)       │
├──────────────┼──────────────────────────────────────────────────┤
│  CONFRONTO   │ Tabella val_acc / test_acc per tutti i modelli   │
│              │ Selezione automatica del migliore                │
├──────────────┼──────────────────────────────────────────────────┤
│  VALUTAZIONE │ Confusion matrix (conteggi assoluti, Blues)      │
│  FINALE      │ Classification report (precision/recall/F1)      │
│              │ Curve di training (loss e accuracy)              │
├──────────────┼──────────────────────────────────────────────────┤
│  REPORT      │ Generazione automatica report HTML con:          │
│              │ configurazione, metriche, grafici, CM            │
└──────────────┴──────────────────────────────────────────────────┘
```

### Strategia di Transfer Learning

Ogni modello viene addestrato in **due fasi distinte**:

1. **Head-only** (`epochs_head = 25`): il backbone pre-addestrato (ImageNet) viene completamente congelato. Si addestra soltanto il nuovo classificatore finale. Questo consente una convergenza rapida senza rischio di distruggere i pesi pre-addestrati.

2. **Fine-tuning** (`epochs_finetune = 30`): vengono sbloccati gli ultimi blocchi del backbone e si riaddestra l'intero modello con un learning rate molto più basso (`lr_finetune = 5e-5`), consentendo un adattamento fine al dominio delle immagini food.

In entrambe le fasi è attivo l'**Early Stopping** (`patience = 5`) e lo scheduler `ReduceLROnPlateau` per evitare overfitting e sprechi computazionali.

---

## 6. Architetture testate

| Modello           | Anno | Tipo                     | Note                                                       |
|-------------------|------|--------------------------|------------------------------------------------------------|
| `vgg16`           | 2014 | CNN classica             | Punto di partenza storico                                  |
| `resnet50`        | 2015 | CNN con skip connections | Introduce i residual blocks                                |
| `efficientnet_b0` | 2019 | CNN scaling uniforme     | Compatto ed efficiente                                     |
| `efficientnet_b4` | 2019 | CNN scaling uniforme     | Upgrade diretto di B0, +3-5% acc. tipica                   |
| `convnext_tiny`   | 2022 | CNN modernizzata         | Ispirata ai Transformer, eccellente rapporto qualità/costo |
| `swin_tiny`       | 2021 | Swin Transformer         | Architettura Transformer con finestre locali               |

La progressione cronologica è intenzionale: rende visibile il miglioramento delle prestazioni man mano che si adottano architetture più recenti.

---

## 7. Tecnologie e dipendenze

| Libreria                | Utilizzo                                            |
|-------------------------|-----------------------------------------------------|
| `torch` / `torchvision` | Framework deep learning, modelli pre-addestrati     |
| `albumentations`        | Pipeline di data augmentation avanzata              |
| `numpy` / `pandas`      | Manipolazione dati e tabelle di confronto           | 
| `matplotlib`            | Visualizzazione grafici, confusion matrix, campioni |
| `scikit-learn`          | Metriche di valutazione (confusion matrix, report)  |
| `pathlib`               | Gestione percorsi multipiattaforma                  |

### Installazione dipendenze

```bash
pip install torch torchvision
pip install albumentations
pip install numpy pandas matplotlib scikit-learn
```

### Requisiti minimi

- Python ≥ 3.9
- RAM ≥ 8 GB (consigliato 16 GB per i modelli più grandi)
- GPU opzionale ma fortemente consigliata (supporta CUDA e Apple MPS)

### Compatibilità hardware e workaround MPS

| Hardware                     | Stato                                |
|------------------------------|--------------------------------------|
| GPU NVIDIA (CUDA)            | Pieno supporto                       |
| CPU                          | Pieno supporto (lento)               |
| Apple Silicon MPS (M1/M2/M3) | Supporto con workaround (vedi sotto) |

Il notebook rileva automaticamente il device disponibile (CUDA → MPS → CPU) e include due workaround per Apple Silicon:

**1. `PYTORCH_ENABLE_MPS_FALLBACK=1`** (impostato prima degli import)  
Abilita il fallback automatico CPU per operazioni non ancora supportate dal backend MPS di PyTorch, come `AdaptiveAvgPool2d` con dimensioni di input non divisibili per l'output size (problema tipico di VGG16 con immagini diverse da 224×224).

**2. Fine-tuning su CPU per ConvNeXt e Swin Transformer**  
ConvNeXt-Tiny e Swin-Tiny presentano un bug noto nel backward pass su MPS ([PyTorch issue #96056](https://github.com/pytorch/pytorch/issues/96056)): durante il fine-tuning, operazioni interne usano `view()` su tensori non contigui in memoria, causando un `RuntimeError`. La pipeline rileva automaticamente questa condizione e sposta il fine-tuning su CPU, riportando poi il modello su MPS per la valutazione finale.

Su **Google Colab con GPU NVIDIA (CUDA)** nessuno dei due workaround si attiva — tutti i modelli girano interamente su GPU.

---

## 8. Come eseguire il notebook

### Su Google Colab (consigliato per la condivisione)

1. Apri il notebook su **[Google Colab](https://colab.research.google.com/)**
2. **Esegui la cella `0) Setup ambiente`** — scarica automaticamente il dataset pubblico e installa `albumentations`
3. **Esegui tutte le celle rimanenti** (`Runtime → Run all`) — la pipeline è completamente automatica

> Il dataset è scaricato da `https://proai-datasets.s3.eu-west-3.amazonaws.com/dataset_food_classification.zip`  
> Se il dataset è già presente nella sessione, il download viene saltato automaticamente.

> Su Google Colab, per evitare la perdita del lavoro in caso di disconnessione, si raccomanda di impostare `cfg = Config(checkpoint_every=3)` e di montare Google Drive per persistere i checkpoint tra sessioni.

### In locale (Jupyter Notebook / JupyterLab)

1. **Clona o scarica** la cartella del progetto
2. **Posiziona il dataset** nella cartella `dataset/` con la struttura `train/val/test/{classe}/`  
   *(oppure lascia che la cella di setup lo scarichi automaticamente)*
3. **Apri** `gourmetai_food_classifier_v1.ipynb` in Jupyter Notebook o JupyterLab
4. **Esegui tutte le celle** (`Run All`) — la pipeline è completamente automatica

Per personalizzare i parametri, modifica la riga di inizializzazione:

```python
# Valori di default (consigliato per la prima esecuzione)
cfg = Config()

# Oppure sovrascrivi solo i parametri che vuoi cambiare, es.:
cfg = Config(batch_size=32, epochs_head=30, patience=8)
cfg = Config(dataset_dir="mio_dataset", lr_finetune=1e-4)
cfg = Config(checkpoint_every=3)    # checkpoint ogni 3 epoch
cfg = Config(checkpoint_every=0)   # disabilita checkpoint periodico
```

### Parametri configurabili (`Config`)

| Parametro          | Default         | Descrizione                                                |
|--------------------|-----------------|------------------------------------------------------------|
| `dataset_dir`      | `"dataset"`     | Percorso cartella dataset                                  |
| `checkpoint_dir`   | `"checkpoints"` | Cartella per i checkpoint                                  |
| `checkpoint_every` | `5`             | Salva checkpoint periodico ogni N epoch (0 = disabilitato) |
| `image_size`       | `224`           | Dimensione delle immagini (px)                             |
| `batch_size`       | `64`            | Dimensione del batch                                       |
| `num_workers`      | `0`             | Worker per il DataLoader (0 = sicuro in Jupyter)           |
| `seed`             | `42`            | Seed per la riproducibilità                                |
| `epochs_head`      | `25`            | Epoche fase head-only                                      |
| `epochs_finetune`  | `30`            | Epoche fase fine-tuning                                    |
| `lr_head`          | `3e-4`          | Learning rate fase head-only                               |
| `lr_finetune`      | `5e-5`          | Learning rate fase fine-tuning                             |
| `weight_decay`     | `1e-3`          | Regolarizzazione L2                                        |
| `patience`         | `5`             | Epoche di tolleranza per Early Stopping                    |

---

## 9. Output prodotti

Al termine dell'esecuzione vengono generati automaticamente:

### Checkpoint

Pesi del modello salvati in `checkpoints/{nome_modello}/`:
- `best_head.pt` — pesi migliori della fase head-only
- `best_finetune.pt` — pesi migliori della fase fine-tuning
- `resume_head.pt` — checkpoint periodico fault tolerance (head)
- `resume_finetune.pt` — checkpoint periodico fault tolerance (fine-tuning)

### Report HTML

Un file `reports/report_YYYY-MM-DD_HH-MM-SS.html` auto-contenuto (immagini incluse in base64) con:
- Configurazione dell'esecuzione
- Statistiche del dataset
- Risultati del modello baseline
- Tabella comparativa di tutti i modelli TL (con evidenziazione del migliore)
- Curve di training (loss e accuracy) per la fase head e fine-tuning
- Confusion matrix finale sul test set
- Classification report completo (precision, recall, F1-score per classe)

### Grafici inline nel notebook

- Curve di training/validation loss e accuracy del modello migliore
- Confusion matrix con conteggi assoluti (colormap Blues)
- Griglia di campioni del dataset con etichette di classe
- Confronto immagine originale vs. versioni aumentate

---

## 10. Suggerimenti per migliorare la precisione

| Intervento                                                | Impatto atteso                   |
|-----------------------------------------------------------|----------------------------------|
| Aumentare `epochs_head` e `epochs_finetune`               | +1–3% accuracy                   |
| Utilizzare `EfficientNet-B7` o `ViT-Base`                 | +3–7% accuracy                   |
| Aggiungere augmentation più aggressiva (CutMix, MixUp)    | +2–4% accuracy                   |
| Raccogliere più immagini per le classi sottorappresentate | +2–5% accuracy                   |
| Test Time Augmentation (TTA)                              | +1–2% accuracy                   |
| Label smoothing nel CrossEntropyLoss                      | Migliora la calibrazione         |
| Learning rate warmup + cosine annealing                   | Convergenza più stabile          |
| Sbloccare blocchi progressivamente durante il fine-tuning | Riduce il rischio di overfitting |

---

## File correlati

| File                                 | Descrizione                               |
|--------------------------------------|-------------------------------------------|
| `gourmetai_food_classifier_v1.ipynb` | Notebook principale (questo progetto)     |
| `config_exp01.md`                    | Configurazione e risultati Esperimento 01 |
| `config_exp02.md`                    | Configurazione e risultati Esperimento 02 |
| `config_exp03.md`                    | Configurazione e risultati Esperimento 03 |

---

*Progetto sviluppato nell'ambito del Master in AI Engineering — Modulo Deep Learning applicato con PyTorch.*
