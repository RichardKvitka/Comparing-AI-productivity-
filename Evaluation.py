from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model_cv(model, X, y, cv=10):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    accuracy, precision, recall, f1 = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy.append(accuracy_score(y_val, y_pred))
        precision.append(precision_score(y_val, y_pred))
        recall.append(recall_score(y_val, y_pred))
        f1.append(f1_score(y_val, y_pred))

    return {
        'Accuracy': np.mean(accuracy),
        'Precision': np.mean(precision),
        'Recall': np.mean(recall),
        'F1 Score': np.mean(f1)
    }
