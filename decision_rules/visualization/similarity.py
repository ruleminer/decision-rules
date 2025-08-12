try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "To use visualization features, install all required packages: "
        "`pip install decision_rules[visualization]`"
    ) from e

import numpy as np
import pandas as pd
from typing import Optional, Union
from decision_rules.similarity.calculate import calculate_rule_similarity
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.similarity import SimilarityType, SimilarityMeasure


def plot_rule_similarity(
    ruleset1: AbstractRuleSet,
    ruleset2: AbstractRuleSet,
    X: pd.DataFrame,
    similarity_type: SimilarityType,
    measure: Optional[SimilarityMeasure] = None,
    top_n_pair: Optional[int] = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    colorscale: str = "Blues",
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Rule Similarity Matrix",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Visualize the rule similarity matrix between two rule sets as a heatmap.

    Parameters
    ----------
    ruleset1 : AbstractRuleSet
        First rule set object.
    ruleset2 : AbstractRuleSet
        Second rule set object.
    X : pandas.DataFrame
        DataFrame providing context for similarity calculation.
    similarity_type : SimilarityType
        Type of similarity: SYNTACTIC or SEMANTIC.
    measure : SimilarityMeasure, optional
        Measure for semantic similarity (required for SEMANTIC).
    top_n_pair : int or None, optional
        If set, show only the top N most similar rule pairs (excluding self-comparisons).
    figsize : tuple, optional
        Figure size (for matplotlib).
    cmap : str, optional
        Colormap for static (matplotlib) plot.
    colorscale : str, optional
        Colorscale for interactive (Plotly) plot.
    ax : matplotlib.axes.Axes or None, optional
        Axis for static plotting. If None, a new axis is created.
    show : bool, default=True
        Whether to display the plot immediately.
    title : str, optional
        Title of the plot.
    save_path : str or None, optional
        If provided, save the plot to this path.
    interactive : bool, default=False
        Whether to use Plotly for an interactive heatmap.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The figure object.
    ax : matplotlib.axes.Axes or None
        Axis object (matplotlib only).
    """

    sim_matrix = calculate_rule_similarity(
        ruleset1, ruleset2, X, similarity_type, measure)
    n1, n2 = sim_matrix.shape
    labels1 = [f"Rule {i+1}" for i in range(n1)]
    labels2 = [f"Rule {j+1}" for j in range(n2)]

    # Select top N pairs (optional)
    if top_n_pair is not None:
        flat_pairs = []
        for i in range(n1):
            for j in range(n2):
                if ruleset1 is ruleset2 and i == j:
                    continue  # skip self-comparisons if comparing the same ruleset
                flat_pairs.append(((i, j), sim_matrix[i, j]))
        sorted_pairs = sorted(flat_pairs, key=lambda x: x[1], reverse=True)
        if len(sorted_pairs) <= top_n_pair:
            selected_pairs = sorted_pairs
        else:
            threshold = sorted_pairs[top_n_pair - 1][1]
            selected_pairs = [p for p in sorted_pairs if p[1] >= threshold]
        selected_rows = sorted(set(i for (i, _), _ in selected_pairs))
        selected_cols = sorted(set(j for (_, j), _ in selected_pairs))
        if not selected_rows or not selected_cols:
            raise ValueError(
                "No rule pairs found above the similarity threshold.")
        sim_matrix = sim_matrix[np.ix_(selected_rows, selected_cols)]
        labels1 = [f"Rule {i+1}" for i in selected_rows]
        labels2 = [f"Rule {j+1}" for j in selected_cols]
        n1, n2 = sim_matrix.shape

    # Interactive (Plotly)
    if interactive:
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=labels2,
            y=labels1,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Similarity", tickvals=[0, 0.5, 1])
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Ruleset 2",
            yaxis_title="Ruleset 1",
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=50)
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    # Static (matplotlib)
    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
    sns.heatmap(sim_matrix, xticklabels=labels2, yticklabels=labels1,
                cmap=cmap, annot=True, fmt=".2f", vmin=0, vmax=1,
                cbar_kws={"ticks": [0, 0.5, 1]}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Ruleset 2")
    ax.set_ylabel("Ruleset 1")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax

