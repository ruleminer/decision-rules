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
from itertools import combinations
from typing import Optional, Union
from decision_rules.core.ruleset import AbstractRuleSet


def _get_top_pair_indices(co_occurrence, top_n) -> list[int]:
    """
    Get the set of unique condition indices involved in the top N co-occurring condition pairs,
    including all ties (i.e., pairs with the same co-occurrence count as the N-th pair).

    Returns
    -------
    list of int
        Sorted list of unique condition indices present in the top N co-occurring pairs.
    """
    n = co_occurrence.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i):
            if not np.isnan(co_occurrence[i, j]):
                pairs.append(((i, j), co_occurrence[i, j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    if len(pairs_sorted) < top_n:
        selected_pairs = pairs_sorted
    else:
        threshold = pairs_sorted[top_n - 1][1]
        selected_pairs = [
            pair for pair in pairs_sorted if pair[1] >= threshold]
    pairs = selected_pairs
    idx_set = set()
    for (i, j), _ in pairs:
        idx_set.add(i)
        idx_set.add(j)
    return sorted(list(idx_set))


def plot_condition_cooccurrence_matrix(
    ruleset: AbstractRuleSet,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    top_n_cooccurrences: Optional[int] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Condition Co-occurrence Matrix",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Plot a heatmap of condition co-occurrences for a given ruleset.

    Parameters
    ----------
    ruleset : AbstractRuleSet
        RuleSet-like object containing .rules and .column_names.
    figsize : tuple, default=(10, 8)
        Figure size (matplotlib only).
    cmap : str, default="Blues"
        Colormap for the heatmap (matplotlib or plotly).
    top_n_cooccurrences : int or None, optional
        If set, display only top N condition pairs (including ties).
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (matplotlib only). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to display the plot immediately.
    title : str, default="Condition Co-occurrence Matrix"
        Plot title.
    save_path : str or None, optional
        If provided, save the plot to this file (static: PNG, interactive: HTML).
    interactive : bool, default=False
        If True, generate an interactive Plotly plot. If False, uses matplotlib.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        Figure object.
    ax : matplotlib.axes.Axes or None
        Axis object (matplotlib only).
    """
    if not hasattr(ruleset, "rules"):
        raise TypeError("Input must be an AbstractRuleSet object.")

    # Build condition co-occurrence matrix
    rule_condition_list = []
    condition_counts = {}
    for rule in ruleset.rules:
        rule_conditions = {cond.to_string(getattr(ruleset, "column_names", []))
                           for cond in getattr(rule.premise, 'subconditions', [])}
        rule_condition_list.append(rule_conditions)
        for cond_str in rule_conditions:
            condition_counts[cond_str] = condition_counts.get(cond_str, 0) + 1

    unique_conditions = sorted(condition_counts.keys())
    n = len(unique_conditions)

    if n == 0:
        raise ValueError("No conditions found in ruleset.")

    co_occurrence = np.zeros((n, n), dtype=float)
    for rule_conditions in rule_condition_list:
        filtered = {c for c in rule_conditions if c in unique_conditions}
        for cond_a, cond_b in combinations(filtered, 2):
            i, j = unique_conditions.index(
                cond_a), unique_conditions.index(cond_b)
            co_occurrence[i, j] += 1
            co_occurrence[j, i] += 1
    np.fill_diagonal(co_occurrence, np.nan)

    if top_n_cooccurrences is not None:
        idxs = _get_top_pair_indices(co_occurrence, top_n_cooccurrences)
        if not idxs:
            raise ValueError("No conditions in top N co-occurring pairs.")
        co_occurrence = co_occurrence[np.ix_(idxs, idxs)]
        unique_conditions = [unique_conditions[i] for i in idxs]
        n = len(unique_conditions)

    vmax = np.nanmax(co_occurrence) if np.nanmax(co_occurrence) > 0 else 1

    # INTERACTIVE (Plotly)
    if interactive:
        z = np.where(np.isnan(co_occurrence), None, co_occurrence)
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=unique_conditions,
                y=unique_conditions,
                colorscale=cmap,
                colorbar=dict(title="Co-occurrence"),
                zmin=0,
                zmax=vmax,
                hoverongaps=False,
                hovertemplate="Condition 1: %{y}<br>Condition 2: %{x}<br>Co-occurrence: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title=title,
            xaxis_title="Conditions",
            yaxis_title="Conditions",
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=50),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    # STATIC (Matplotlib)
    annot_matrix = np.empty_like(co_occurrence, dtype=object)
    for i in range(n):
        for j in range(n):
            annot_matrix[i, j] = "" if np.isnan(
                co_occurrence[i, j]) else f"{int(co_occurrence[i, j])}"

    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
    sns.heatmap(
        co_occurrence,
        mask=np.triu(np.ones_like(co_occurrence, dtype=bool)),
        xticklabels=unique_conditions,
        yticklabels=unique_conditions,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        cbar_kws={"ticks": []},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Conditions")
    ax.set_ylabel("Conditions")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


def plot_attribute_cooccurrence_matrix(
    ruleset: AbstractRuleSet,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Attribute Co-occurrence Matrix",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    top_n_cooccurrences: Optional[int] = None,
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Plot a heatmap of attribute co-occurrences for a given ruleset.

    Parameters
    ----------
    ruleset : AbstractRuleSet
        RuleSet-like object containing .rules and .column_names.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (matplotlib only). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to display the plot immediately.
    title : str, default="Attribute Co-occurrence Matrix"
        Plot title.
    figsize : tuple, default=(10, 8)
        Figure size (matplotlib only).
    cmap : str, default="Blues"
        Colormap for the heatmap (matplotlib or plotly).
    top_n_cooccurrences : int or None, optional
        If set, display only top N attribute pairs (including ties).
    save_path : str or None, optional
        If provided, save the plot to this file (static: PNG, interactive: HTML).
    interactive : bool, default=False
        If True, generate an interactive Plotly plot. If False, uses matplotlib.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        Figure object.
    ax : matplotlib.axes.Axes or None
        Axis object (matplotlib only).
    """
    if not hasattr(ruleset, "rules"):
        raise TypeError("Input must be an AbstractRuleSet object.")

    rule_attribute_list = []
    attribute_counts = {}
    for rule in ruleset.rules:
        rule_attrs = {getattr(ruleset, "column_names", [])[cond.column_index]
                      for cond in getattr(rule.premise, 'subconditions', [])}
        rule_attribute_list.append(rule_attrs)
        for attr in rule_attrs:
            attribute_counts[attr] = attribute_counts.get(attr, 0) + 1

    unique_attributes = sorted(attribute_counts.keys())
    n = len(unique_attributes)

    if n == 0:
        raise ValueError("No attributes found in ruleset.")

    co_occurrence = np.zeros((n, n), dtype=float)
    for rule_attrs in rule_attribute_list:
        filtered = {a for a in rule_attrs if a in unique_attributes}
        for attr_a, attr_b in combinations(filtered, 2):
            i, j = unique_attributes.index(
                attr_a), unique_attributes.index(attr_b)
            co_occurrence[i, j] += 1
            co_occurrence[j, i] += 1
    np.fill_diagonal(co_occurrence, np.nan)

    if top_n_cooccurrences is not None:
        idxs = _get_top_pair_indices(co_occurrence, top_n_cooccurrences)
        if not idxs:
            raise ValueError("No attributes in top N co-occurring pairs.")
        co_occurrence = co_occurrence[np.ix_(idxs, idxs)]
        unique_attributes = [unique_attributes[i] for i in idxs]
        n = len(unique_attributes)

    vmax = np.nanmax(co_occurrence) if np.nanmax(co_occurrence) > 0 else 1

    # INTERACTIVE (Plotly)
    if interactive:
        z = np.where(np.isnan(co_occurrence), None, co_occurrence)
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=unique_attributes,
                y=unique_attributes,
                colorscale=cmap,
                colorbar=dict(title="Co-occurrence"),
                zmin=0,
                zmax=vmax,
                hoverongaps=False,
                hovertemplate="Attribute 1: %{y}<br>Attribute 2: %{x}<br>Co-occurrence: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title=title,
            xaxis_title="Attributes",
            yaxis_title="Attributes",
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=50),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    # STATIC (Matplotlib)
    annot_matrix = np.empty_like(co_occurrence, dtype=object)
    for i in range(n):
        for j in range(n):
            annot_matrix[i, j] = "" if np.isnan(
                co_occurrence[i, j]) else f"{int(co_occurrence[i, j])}"

    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
    sns.heatmap(
        co_occurrence,
        mask=np.triu(np.ones_like(co_occurrence, dtype=bool)),
        xticklabels=unique_attributes,
        yticklabels=unique_attributes,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        cbar_kws={"ticks": []},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Attributes")
    ax.set_ylabel("Attributes")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax
