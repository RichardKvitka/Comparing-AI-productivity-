from sklearn.tree import DecisionTreeClassifier
from Evaluation import evaluate_model_cv

def grid_search_tree(X, y):
    criterions = ['gini', 'entropy']
    depths = [3, 5, 10, 15, 20]

    best_score = 0
    best_params = {}

    for c in criterions:
        for d in depths:
            model = DecisionTreeClassifier(criterion=c, max_depth=d, random_state=42)
            metrics = evaluate_model_cv(model, X, y)
            if metrics['F1 Score'] > best_score:
                best_score = metrics['F1 Score']
                best_params = {
                    'criterion': c,
                    'max_depth': d,
                    'metrics': metrics
                }
    return best_params
