import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections.abc import Sequence
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from typing import Optional, Union, Any
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.regression.rule import RegressionRule


def plot_rule_target_histogram(
    rules: Union[
        RegressionRuleSet,
        RegressionRule,
        list[RegressionRule]
    ],
    X: pd.DataFrame,
    y: Union[pd.Series, Any],
    bins: Union[int, str, Sequence] = "auto",
    base_color: Optional[str] = None,
    covered_colors: Optional[list] = None,
    alpha_base: float = 0.35,
    alpha_covered: float = 0.85,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Histogram of target (y) and covered examples",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Visualize the histogram of the target variable and overlay histograms for examples covered by rules.

    Parameters
    ----------
    rules : RegressionRuleSet, RegressionRule, or list of RegressionRule
        Rule set object, single rule, or list of rules (must support .premise.covered_mask(X)).
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix.
    y : array-like
        Target variable.
    bins : int, str or sequence, optional
        Number of histogram bins or strategy, passed to numpy.histogram_bin_edges.
    base_color : str or None, optional
        Color of the base histogram (all examples).
    covered_colors : list or None, optional
        List of colors for covered sets. If None, a palette is used.
    alpha_base : float, default=0.35
        Transparency of the base histogram.
    alpha_covered : float, default=0.85
        Transparency of the covered histograms.
    ax : matplotlib.axes.Axes or None, optional
        Axis for plotting (static mode). If None, a new axis is created.
    show : bool, default=True
        Whether to display the plot immediately (static/interactive).
    title : str, optional
        Title for the plot.
    save_path : str or None, optional
        Path to save the plot (PNG for static, HTML for interactive).
    interactive : bool, default=False
        Whether to use Plotly for an interactive plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The figure object.
    ax : matplotlib.axes.Axes or None
        Axis object (matplotlib only).
    """
    # Normalize rules input
    if hasattr(rules, "rules"):
        rules = rules.rules
    elif isinstance(rules, (list, tuple)):
        rules = list(rules)
    else:
        rules = [rules]

    y = np.asarray(y)
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    bin_edges = np.histogram_bin_edges(y, bins=bins)

    if base_color is None:
        base_color = "#222222"
    if covered_colors is None:
        palette = sns.color_palette("Set1", n_colors=len(rules))
        covered_colors = palette.as_hex() if hasattr(palette, "as_hex") else palette

    if interactive:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y, xbins=dict(
                start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
            marker_color=base_color, opacity=alpha_base, name="All examples"
        ))
        for i, rule in enumerate(rules):
            mask = rule.premise.covered_mask(X_np)
            y_covered = y[mask]
            if len(y_covered) > 0:
                fig.add_trace(go.Histogram(
                    x=y_covered, xbins=dict(
                        start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])),
                    marker_color=covered_colors[i % len(covered_colors)],
                    opacity=alpha_covered,
                    name=f"Rule {i}: {str(rule)[:40]}..."
                ))
        mean = np.mean(y)
        std = np.std(y)
        fig.add_vline(x=mean, line_dash="dash", line_color="black",
                      annotation_text="mean", annotation_position="top")
        fig.add_vline(x=mean + std, line_dash="dot", line_color="black",
                      annotation_text="+1 std", annotation_position="top right")
        fig.add_vline(x=mean - std, line_dash="dot", line_color="black",
                      annotation_text="-1 std", annotation_position="top left")
        fig.update_layout(
            barmode='overlay',
            xaxis_title="Target (y)",
            yaxis_title="Count",
            title=title,
            legend=dict(itemsizing='constant'),
            width=1000,
            height=500
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    # Static matplotlib version
    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)
    ax.hist(y, bins=bin_edges, color=base_color, alpha=alpha_base,
            label="All examples", edgecolor='black', linewidth=0.7)
    for i, rule in enumerate(rules):
        mask = rule.premise.covered_mask(X_np)
        y_covered = y[mask]
        if len(y_covered) > 0:
            ax.hist(
                y_covered, bins=bin_edges,
                color=covered_colors[i % len(covered_colors)],
                alpha=alpha_covered,
                label=f"Rule {i}: {str(rule)[:40]}...",
                edgecolor='black', linewidth=0.7
            )
    mean = np.mean(y)
    std = np.std(y)
    ax.axvline(mean, color="black", linestyle="--", linewidth=2, label="mean")
    ax.axvline(mean + std, color="black", linestyle=":",
               linewidth=1.5, label="+/- 1 std")
    ax.axvline(mean - std, color="black", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Target (y)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax
