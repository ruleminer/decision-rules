import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import plotly.graph_objects as go
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Union, Any
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.classification.rule import ClassificationRule


def rgb_tuple_to_rgba(rgb, alpha=1.0):
    """Convert an RGB tuple (0-1 floats) to an RGBA CSS string for Plotly."""
    return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"


def plot_rule_coverage_distribution(
    rules: Union[
        ClassificationRuleSet,
        ClassificationRule,
        list[ClassificationRule]
    ],
    X: pd.DataFrame,
    y: Union[pd.Series, Any],
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Rule coverage per class",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Plots the distribution of rule coverage across classes for a given set of classification rules.

    Parameters
    ----------
    rules : ClassificationRuleSet, ClassificationRule, or list of ClassificationRule
        Set of rules to visualize. Can be a ClassificationRuleSet, a single ClassificationRule, or a list of ClassificationRule objects.
    X : pandas.DataFrame
        Feature matrix (samples x features) to evaluate rule coverage.
    y : array-like or pandas.Series
        Ground truth class labels for each sample in X.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (only used if interactive=False). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to immediately display the plot (calls plt.show() or fig.show()).
    title : str, default="Rule coverage per class"
        Title of the plot.
    save_path : str or None, optional
        If provided, saves the plot to this path. For static plots: PNG; for interactive: HTML.
    interactive : bool, default=False
        If True, uses Plotly for interactive plotting. If False, uses Matplotlib for static plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The created figure object (Matplotlib or Plotly, depending on `interactive`).
    ax : matplotlib.axes.Axes or None
        The axis object (only for Matplotlib; None for Plotly).
    """
    # Normalize rules input
    if hasattr(rules, "rules"):
        rules = rules.rules
    elif isinstance(rules, (list, tuple)):
        rules = list(rules)
    else:
        rules = [rules]

    classes = np.unique(y)
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    n_classes = len(classes)
    n_rules = len(rules)

    # Compute coverage matrix: rules x classes
    coverage_matrix = np.zeros((n_rules, n_classes), dtype=int)
    positive_class_indices = [
        class_to_index[rule.conclusion.value] for rule in rules]

    for i, rule in enumerate(rules):
        covered = rule.premise.covered_mask(X.to_numpy())
        for cls in classes:
            class_mask = (y == cls) & covered
            coverage_matrix[i, class_to_index[cls]] = np.sum(class_mask)

    rule_labels = [f"R{i+1}" for i in range(n_rules)]
    colors = cm.get_cmap("tab10").colors[:n_classes]

    if interactive:
        fig = go.Figure()
        for j, cls in enumerate(classes):
            y_values = coverage_matrix[:, j]
            bar_colors = [
                rgb_tuple_to_rgba(
                    colors[j], 1.0 if positive_class_indices[i] == j else 0.3)
                for i in range(n_rules)
            ]
            fig.add_bar(
                x=rule_labels,
                y=y_values,
                name=f"Class {cls}",
                marker=dict(color=bar_colors),
                hovertemplate=(
                    "Rule: %{x}<br>Class: " + str(cls) +
                    "<br>Count: %{y}<extra></extra>"
                ),
            )
        fig.update_layout(
            title=title,
            xaxis_title="Rule",
            yaxis_title="Count",
            barmode="group",
            bargap=0.15,
            legend_title="Class",
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    # Matplotlib static plotting
    fig, ax = plt.subplots(figsize=(max(4, n_rules), 5)
                           ) if ax is None else (ax.figure, ax)
    ind = np.arange(n_rules)
    width = 0.8 / n_classes
    legend_handles = []

    for j, cls in enumerate(classes):
        for i in range(n_rules):
            value = coverage_matrix[i, j]
            alpha = 1.0 if positive_class_indices[i] == j else 0.3
            ax.bar(
                ind[i] - 0.4 + j * width + width / 2,
                value,
                width=width,
                color=colors[j],
                alpha=alpha,
                label=None,  # Legends handled below
            )
        # Legend: always full color for each class
        handle = plt.Line2D(
            [0], [0],
            marker='s', color='w',
            label=f"Class {cls}",
            markerfacecolor=colors[j],
            markersize=10,
            linestyle='None'
        )
        legend_handles.append(handle)

    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(rule_labels)
    ax.set_ylim(0, max(1, coverage_matrix.max() + 5))
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax
