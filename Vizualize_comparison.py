import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(default_metrics, tuned_metrics, model_name):
    labels = list(default_metrics.keys())
    default_scores = [default_metrics[m] for m in labels]
    tuned_scores = [tuned_metrics[m] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, default_scores, width, label='Baseline', color='lightcoral')
    rects2 = ax.bar(x + width/2, tuned_scores, width, label='Tuned', color='mediumseagreen')

    ax.set_ylabel('Metric Value')
    ax.set_title(f'Comparison of Metrics for {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()
