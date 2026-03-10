# ml-classification-ufc-kaggle-dataset
ML pipeline for classifying UFC matches. Prevents data leakage by separating stateless and stateful cleaning, and uses differential features against bias. Avoids overfitting by preferring Logistic Regression (L1 for feature selection, L2 with TimeSeriesSplit) to Ensemble models. Extracts a real signal from noise (ROC-AUC ~0.615).
