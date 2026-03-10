# ML-classification-ufc-kaggle-dataset
ML pipeline for classifying UFC matches. Prevents data leakage by separating stateless and stateful cleaning, and uses differential features against bias. Avoids overfitting by preferring Logistic Regression (L1 for feature selection, L2 with TimeSeriesSplit) to Ensemble models. Extracts a real signal from noise (ROC-AUC ~0.615).


# UFC Match Outcome Prediction: A Quantitative and Architectural Analysis

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)

## Abstract
Questo progetto implementa una pipeline *end-to-end* di Machine Learning per la classificazione binaria dell'esito degli incontri della Ultimate Fighting Championship (UFC). 
Affrontando un dominio caratterizzato da estrema varianza stocastica e rumore bianco, lo studio si concentra sulla corretta astrazione ingegneristica dello spazio delle *feature* e sulla prevenzione rigorosa del **Data Leakage** temporale, stabilendo il limite teorico dell'informazione estraibile da vettori puramente numerici pre-match.

## Architettura dei Dati e Preprocessing

L'ingegnerizzazione dei dati è stata progettata per garantire l'assoluta integrità della validazione *Out-of-Sample* e *Out-of-Time*:
- **Stateless & Stateful Transformations:** Le operazioni di purificazione del dominio (rimozione pareggi, *zero-imputation* per i debuttanti) sono eseguite come trasformazioni *stateless*. L'imputazione dei dati biometrici mancanti (mediane fisiche) è trattata come operazione *stateful*, appresa rigorosamente sul *Train Set* post Chrono-Split e proiettata sul *Test Set*.
- **Symmetry Feature Engineering:** Le statistiche assolute dei lottatori sono state collassate in feature differenziali vettoriali per neutralizzare il *Red Corner Bias* indotto dalle scelte tattiche dei *matchmaker*.

## Model Selection & Tuning

Il benchmarking iniziale (valutato tramite `TimeSeriesSplit`) ha dimostrato che architetture non-lineari ad alta varianza (Random Forest, XGBoost) tendono a collassare sulla classe maggioritaria (*Overfitting* sul rumore). La strategia definitiva adotta un approccio lineare regolarizzato:
1. **Embedded Feature Selection (Lasso L1):** Induzione matematica di sparsità per decimare lo spazio vettoriale (da ~84 a < 20 variabili), eliminando feature stocastiche e multicollinearità.
2. **Logistic Regression (Ridge L2):** Il classificatore primario, sottoposto a un'ottimizzazione esaustiva degli iperparametri (GridSearchCV) con forte costrizione geometrica sui pesi residui (`C=0.01`).

## Risultati e Metriche (Test Set)

Il modello è stato valutato su una *Naive Baseline* (Zero-R / Dummy Classifier) per quantificare il reale *Lift* predittivo:
- **Naive Baseline ROC-AUC:** 0.5000 (Casualità)
- **Modello Ottimizzato ROC-AUC:** **0.6154**

Il *Generalization Gap* fisiologico tra i risultati in Cross-Validation e quelli sul Test Set puro conferma l'assenza di leakage e quantifica l'evoluzione del *metagioco* delle MMA negli anni recenti.

## Struttura della Repository

```text
├── data/
│   └── data.csv                  # Dataset originale (~5 MB)
├── notebooks/
│   └── 01_ufc_match_prediction_pipeline.ipynb  # Pipeline analitica completa
├── requirements.txt              # Dipendenze dell'ambiente Python
├── LICENSE                       # Licenza MIT
└── README.md                     # Project overview
```

## Riproducibilità
Clonare la repository e installare le dipendenze per eseguire il notebook:

```bash
git clone https://github.com/Francesco002511/ufc-predictive-modeling.git
cd ufc-predictive-modeling
pip install -r requirements.txt
```

## Licenza
Distribuito sotto licenza MIT. Vedi il file `LICENSE` per maggiori informazioni.
