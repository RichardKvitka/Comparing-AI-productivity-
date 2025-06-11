from sklearn.neural_network import MLPClassifier
from Evaluation import evaluate_model_cv


def grid_search_mlp(X, y):
    hidden_layers = [(50,), (100,), (100, 50), (50, 50)]
    activations = ['relu', 'tanh']
    learning_rates = ['constant', 'adaptive']

    best_score = 0
    best_params = {}

    for h in hidden_layers:
        for a in activations:
            for lr in learning_rates:
                model = MLPClassifier(hidden_layer_sizes=h,
                                      activation=a,
                                      learning_rate=lr,
                                      max_iter=1000,
                                      random_state=42)
                metrics = evaluate_model_cv(model, X, y)
                if metrics['F1 Score'] > best_score:
                    best_score = metrics['F1 Score']
                    best_params = {
                        'hidden_layer_sizes': h,
                        'activation': a,
                        'learning_rate': lr,
                        'metrics': metrics
                    }
    return best_params
