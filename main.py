from Preprocessing import load_and_preprocess
from Evaluation import evaluate_model_cv
from MLP import grid_search_mlp
from DecisionTree import grid_search_tree
from Vizualize_comparison import plot_comparison
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import json

X, y = load_and_preprocess()

# Basic Models
default_mlp = MLPClassifier(random_state=42)
default_tree = DecisionTreeClassifier(random_state=42)

default_mlp_metrics = evaluate_model_cv(default_mlp, X, y)
default_tree_metrics = evaluate_model_cv(default_tree, X, y)

# Optimized models
best_mlp = grid_search_mlp(X, y)
best_tree = grid_search_tree(X, y)

# Visualisation
plot_comparison(default_mlp_metrics, best_mlp['metrics'], "MLPClassifier")
plot_comparison(default_tree_metrics, best_tree['metrics'], "DecisionTreeClassifier")


with open("best_mlp.json", "w") as f:
    json.dump(best_mlp, f, indent=4)

with open("best_tree.json", "w") as f:
    json.dump(best_tree, f, indent=4)


print("Default MLP:", default_mlp_metrics)
print("Tuned MLP:", best_mlp["metrics"])
print("Default Tree:", default_tree_metrics)
print("Tuned Tree:", best_tree["metrics"])