try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "To use visualization features, install all required packages: "
        "`pip install decision_rules[visualization]`"
    ) from e

import pandas as pd
from typing import Optional, Union, Any
from decision_rules.core.ruleset import AbstractRuleSet


def plot_condition_importance(
    ruleset: AbstractRuleSet,
    X: pd.DataFrame,
    y: Union[pd.Series, Any],
    measure: Optional[Any] = None,
    max_conditions: int = 10,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Condition Importance",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Visualizes the importance of individual conditions in a rule set for classification, regression, or survival tasks.

    Parameters
    ----------
    ruleset : AbstractRuleSet
        RuleSet-like object implementing calculate_condition_importances(X, y, measure=...).
    X : pandas.DataFrame
        Feature matrix (samples x features) for evaluating condition importance.
    y : pandas.Series or array-like
        Target values (labels or regression targets).
    measure : callable or any, optional
        The measure function or parameter used in calculating condition importances.
    max_conditions : int, default=10
        Maximum number of most important conditions to display.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (matplotlib only). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to display the plot immediately (calls plt.show() or fig.show()).
    title : str, default="Condition Importance"
        Title of the plot.
    save_path : str or None, optional
        If provided, saves the plot to this path. For static plots: PNG; for interactive: HTML.
    interactive : bool, default=False
        If True, generates an interactive Plotly plot. If False, uses Matplotlib.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The created figure object (Matplotlib or Plotly, depending on `interactive`).
    ax : matplotlib.axes.Axes or None
        The axis object (only for Matplotlib; None for Plotly).
    """
    if hasattr(ruleset, "calculate_condition_importances"):
        if measure is not None:
            condition_importances = ruleset.calculate_condition_importances(
                X, y, measure=measure)
        else:
            condition_importances = ruleset.calculate_condition_importances(
                X, y)
    else:
        raise TypeError(
            "Input must be a AbstractRuleset object supporting calculate_condition_importances()."
        )

    if isinstance(condition_importances, list):
        condition_importances = {"Condition Importance": condition_importances}

    keys = list(condition_importances.keys())
    n_keys = len(keys)

    if interactive:
        fig = go.Figure()
        for idx, label in enumerate(keys):
            data = sorted(condition_importances[label], key=lambda x: x["importance"], reverse=True)[
                :max_conditions]
            conditions = [item["condition"] for item in data]
            importances = [item["importance"] for item in data]
            fig.add_trace(go.Bar(
                x=importances,
                y=conditions,
                orientation='h',
                name=label
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Condition",
            yaxis=dict(autorange='reversed'),
            barmode='group'
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    fig, axes = plt.subplots(nrows=n_keys, ncols=1, figsize=(
        8, 4 * n_keys), squeeze=False) if ax is None else (ax.figure, [ax])
    axes = axes.flatten()

    all_importances = []
    for label in keys:
        data = sorted(condition_importances[label], key=lambda x: x["importance"], reverse=True)[
            :max_conditions]
        all_importances.extend([item["importance"] for item in data])
    if all_importances:
        imp_min, imp_max = min(all_importances), max(all_importances)
        xmax = imp_max + 0.05 * \
            (imp_max - imp_min) if imp_max > imp_min else imp_max + \
            0.1 * abs(imp_max) or 0.1
    else:
        imp_min, xmax = 0, 0.1

    for idx, label in enumerate(keys):
        data = sorted(condition_importances[label], key=lambda x: x["importance"], reverse=True)[
            :max_conditions]
        conditions = [item["condition"] for item in data]
        importances = [item["importance"] for item in data]
        axes[idx].barh(conditions, importances, color='#435272', alpha=0.8)
        axes[idx].set_xlabel("Importance")
        axes[idx].set_ylabel("Condition")
        axes[idx].set_title(label if n_keys > 1 else title)
        axes[idx].invert_yaxis()
        axes[idx].xaxis.grid(True, linestyle='--', alpha=0.6)
        axes[idx].set_xlim(imp_min, xmax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, axes if len(axes) > 1 else axes[0]


def plot_attribute_importance(
    ruleset: AbstractRuleSet,
    X: pd.DataFrame,
    y: Union[pd.Series, Any],
    measure: Optional[Any] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Attribute Importance",
    max_attributes: int = 10,
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Visualizes the importance of attributes (features) in a rule set for classification, regression, or survival tasks.

    Parameters
    ----------
    ruleset : AbstractRuleSet
        RuleSet-like object implementing:
            - calculate_condition_importances(X, y, measure=...)
            - calculate_attribute_importances(condition_importances)
    X : pandas.DataFrame
        Feature matrix (samples x features) for evaluating attribute importance.
    y : pandas.Series or array-like
        Target values (labels or regression targets).
    measure : callable or any, optional
        The measure function or parameter used in calculating condition importances.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (matplotlib only). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to display the plot immediately (calls plt.show() or fig.show()).
    title : str, default="Attribute Importance"
        Title of the plot.
    max_attributes : int, default=10
        Maximum number of most important attributes to display.
    save_path : str or None, optional
        If provided, saves the plot to this path. For static plots: PNG; for interactive: HTML.
    interactive : bool, default=False
        If True, generates an interactive Plotly plot. If False, uses Matplotlib.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The created figure object (Matplotlib or Plotly, depending on `interactive`).
    ax : matplotlib.axes.Axes or None
        The axis object (only for Matplotlib; None for Plotly).
    """
    if measure is not None:
        condition_importances = ruleset.calculate_condition_importances(
            X, y, measure=measure)
    else:
        condition_importances = ruleset.calculate_condition_importances(X, y)

    attribute_importances = ruleset.calculate_attribute_importances(
        condition_importances)

    if isinstance(attribute_importances, list):
        attribute_importances = {"Attribute Importance": attribute_importances}
    elif isinstance(attribute_importances, dict):
        if all(isinstance(val, (int, float)) for val in attribute_importances.values()):
            attribute_importances = {
                "Attribute Importance": [
                    {"attribute": k, "importance": v} for k, v in attribute_importances.items()
                ]
            }
        else:
            for key in attribute_importances:
                inner = attribute_importances[key]
                if isinstance(inner, dict):
                    attribute_importances[key] = [
                        {"attribute": k, "importance": v} for k, v in inner.items()
                    ]

    keys = list(attribute_importances.keys())
    n_keys = len(keys)

    if interactive:
        fig = go.Figure()
        for idx, label in enumerate(keys):
            data = sorted(attribute_importances[label], key=lambda x: x["importance"], reverse=True)[
                :max_attributes]
            attributes = [item["attribute"] for item in data]
            importances = [item["importance"] for item in data]
            fig.add_trace(go.Bar(
                x=importances,
                y=attributes,
                orientation='h',
                name=label,  # każdy label osobny kolor
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Attribute",
            yaxis=dict(autorange='reversed'),
            barmode='group'
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    fig, axes = plt.subplots(nrows=n_keys, ncols=1, figsize=(
        8, 4 * n_keys), squeeze=False) if ax is None else (ax.figure, [ax])
    axes = axes.flatten()

    all_importances = []
    for label in keys:
        data = sorted(attribute_importances[label], key=lambda x: x["importance"], reverse=True)[
            :max_attributes]
        all_importances.extend([item["importance"] for item in data])
    if all_importances:
        imp_min, imp_max = min(all_importances), max(all_importances)
        xmax = imp_max + 0.05 * \
            (imp_max - imp_min) if imp_max > imp_min else imp_max + \
            0.1 * abs(imp_max) or 0.1
    else:
        imp_min, xmax = 0, 0.1

    for idx, label in enumerate(keys):
        data = sorted(attribute_importances[label], key=lambda x: x["importance"], reverse=True)[
            :max_attributes]
        attributes = [item["attribute"] for item in data]
        importances = [item["importance"] for item in data]
        axes[idx].barh(attributes, importances, color='#D15E2E', alpha=0.8)
        axes[idx].set_xlabel("Importance")
        axes[idx].set_ylabel("Attribute")
        axes[idx].set_title(label if n_keys > 1 else title)
        axes[idx].invert_yaxis()
        axes[idx].xaxis.grid(True, linestyle='--', alpha=0.6)
        axes[idx].set_xlim(imp_min, xmax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, axes if len(axes) > 1 else axes[0]
