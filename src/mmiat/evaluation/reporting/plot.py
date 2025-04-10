import matplotlib.pyplot as plt

from typing import Any, Optional, Union

def create_mia_metric_report(
        metrics, 
        attack_name: Optional[Union[str, None]]=None,
        extras: Optional[list]=[]) -> None:
    """
    Generates and displays a ROC curve plot for a given set of metrics from a membership inference attack.
    
    Args:
        metrics : dict
            A dictionary containing the false positive rates ('fpr'), true positive rates ('tpr'), and optionally
            other metrics such as 'auc' and 'accuracy'.
        attack_name : str, optional
            The name of the attack to be displayed in the plot title and legend. Defaults to "Attack" if not provided.
        extras : list, optional
            A list of additional options for the plot. Supported options include:
            - "loglog": Sets both x and y axes to a logarithmic scale.
            - "auc": Displays the Area Under the Curve (AUC) value on the plot.
            - "accuracy": Displays the accuracy value on the plot.
            - "allmetrics": Displays all available metrics (AUC and accuracy) on the plot.
        
    Returns:
        None
    """
    if attack_name is None:
        attack_name = "Attack"
    
    fig, ax = plt.subplots()
    if "loglog" in extras:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.2)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    ax.plot(metrics["fpr"], metrics["tpr"], label="Attack" if attack_name is None else attack_name)
    ax.set_xlim([1e-5, 1])
    ax.set_ylim([1e-5, 1])
    ax.legend()
    ax.set(xlabel="False Positive Rate", 
        ylabel="True Positive Rate", 
        title=f"ROC Curve for {attack_name}")
    
    metrics_str = []
    if "auc" in extras or "allmetrics" in extras:
        metrics_str.append(
            f"AUC: {metrics['auc']:.2f}"
        )
    if "accuracy" in extras or "allmetrics" in extras:
        metrics_str.append(
            f"Accuracy: {metrics['accuracy']:.2%}"
        )
    if len(metrics_str) > 0:
        # Display metrics values in the middle of the plot in a white square
        metrics_text = "\n".join(metrics_str)
        ax.text(0.75, 0.25, metrics_text, fontsize=12, ha='center', va='center',
            transform=ax.transAxes,  # Use axes coordinates
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    plt.tight_layout()
    plt.show()

def show_various_mia_attack_comparison(
        attacks_metrics: dict[str, dict[str, Any]], 
        extras: list=[]) -> None:
    """
    Plots a comparison of various metrics for multiple Membership Inference Attack (MIA).

    Args:
        attacks_metrics (dict[str, dict[str, Any]]): A dictionary where keys are attack names 
            and values are dictionaries containing metrics such as 'fpr' (False Positive Rate) 
            and 'tpr' (True Positive Rate).
        extras (list, optional): A list of extra options for the plot. Supported options:
            - "loglog": Sets both x and y axes to logarithmic scale.
            - "accuracy": Appends accuracy information to the legend labels.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if "loglog" in extras:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    for attack_name, metrics in attacks_metrics.items():
        legend_label = f"{attack_name}"
        if "accuracy" in extras:
            legend_label +=  f" (acc={metrics['accuracy']:.1%})"
        ax.plot(metrics["fpr"], metrics["tpr"], label=legend_label)
    ax.set_xlim([1e-3, 1])
    ax.set_ylim([1e-3, 1])
    # Add legend to the plot, on the right edge, bottom aligned
    legend = ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.2, 0.0))
    legend.get_frame().set_alpha(None)
    ax.set(xlabel="False Positive Rate", 
        ylabel="True Positive Rate")
    plt.tight_layout()
    plt.show()