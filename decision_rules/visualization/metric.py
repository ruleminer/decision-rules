import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Union, Any
from decision_rules.core.ruleset import AbstractRuleSet


def plot_ruleset_metrics(
    ruleset: AbstractRuleSet,
    X: pd.DataFrame,
    y: Union[pd.Series, Any],
    x_metric: str,
    y_metric: str,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[Any, None]
]:
    """
    Generates a 2D scatter plot of selected rule metrics for a ruleset.

    Supports both interactive Plotly and static matplotlib backends.

    Parameters
    ----------
    ruleset : AbstractRuleSet
        RuleSet-like object exposing .calculate_rules_metrics(X, y) and .rules.
    X : pandas.DataFrame
        Feature matrix (samples x features).
    y : pandas.Series or array-like
        Target vector (ground truth labels or values).
    x_metric : str
        Name of the metric to plot on the x-axis (must be a key in the metrics dict).
    y_metric : str
        Name of the metric to plot on the y-axis (must be a key in the metrics dict).
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (matplotlib only). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to display the plot immediately (calls plt.show() or fig.show()).
    title : str or None, optional
        Plot title. If None, a default title is generated.
    save_path : str or None, optional
        If provided, saves the plot to this file (static: PNG, interactive: HTML).
    interactive : bool, default=False
        If True, uses Plotly for an interactive plot. If False, uses matplotlib.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The created figure object (Matplotlib or Plotly, depending on `interactive`).
    ax : matplotlib.axes.Axes or None
        The axis object (only for Matplotlib; None for Plotly).
    """
    if not hasattr(ruleset, "calculate_rules_metrics"):
        raise TypeError(
            "Input must be a AbstractRuleset object supporting a calculate_rules_metrics(X, y) method.")

    metrics = ruleset.calculate_rules_metrics(X, y)
    rule_ids = list(metrics.keys())
    # Extract rule texts and decisions
    rule_texts = []
    rule_decisions = []
    for rid in rule_ids:
        rule_obj = next((r for r in getattr(ruleset, "rules", [])
                        if getattr(r, "uuid", None) == rid), None)
        if rule_obj is not None:
            rule_texts.append(str(rule_obj))
            rule_decisions.append(getattr(rule_obj.conclusion, "value", None))
        else:
            rule_texts.append(rid)
            rule_decisions.append(None)

    # Retrieve values for the selected metrics
    x_values = [metrics[rid].get(x_metric, np.nan) for rid in rule_ids]
    y_values = [metrics[rid].get(y_metric, np.nan) for rid in rule_ids]

    # Prepare DataFrame and drop invalid rows
    data = pd.DataFrame({
        "x": x_values,
        "y": y_values,
        "decision": rule_decisions,
        "rule": rule_texts
    }).dropna(subset=["x", "y"])

    if data.empty:
        raise ValueError(
            "No data points to plot. Check metric names or rule metrics.")

    plot_title = title if title is not None else f"Relationship between {x_metric.capitalize()} and {y_metric.capitalize()}"

    if interactive:
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color="decision",
            hover_data=["rule"],
            title=plot_title,
            labels={"x": x_metric.capitalize(), "y": y_metric.capitalize()}
        )
        fig.update_traces(
            marker=dict(size=12, line=dict(width=1, color="#131720")),
            selector=dict(mode="markers")
        )
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    # Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)
    decisions = pd.Series(data["decision"])

    if pd.api.types.is_numeric_dtype(decisions):
        decisions = pd.to_numeric(decisions, errors="coerce")
        finite_mask = np.isfinite(decisions)
        if not finite_mask.all():
            finite_vals = decisions[finite_mask]
            max_finite = finite_vals.max() if not finite_vals.empty else 0
            decisions[~finite_mask] = max_finite + 10

        sc = ax.scatter(
            data["x"], data["y"],
            c=decisions,
            cmap="viridis",
            s=75, alpha=0.7, edgecolors="k"
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Decision")
    else:
        categorical_decisions = decisions.fillna("Unknown").astype("category")
        cat_colors = {cat: plt.get_cmap("tab10")(i)
                      for i, cat in enumerate(categorical_decisions.cat.categories)}
        colors = [cat_colors[val] for val in categorical_decisions]
        ax.scatter(data["x"], data["y"], color=colors,
                   s=75, alpha=0.7, edgecolors="k")
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                              markerfacecolor=col, markersize=8, markeredgecolor="k")
                   for cat, col in cat_colors.items()]
        ax.legend(handles=handles, title="Decision")

    ax.set_xlabel(x_metric.capitalize())
    ax.set_ylabel(y_metric.capitalize())
    ax.set_title(plot_title)
    x_range = data["x"].max() - data["x"].min()
    y_range = data["y"].max() - data["y"].min()
    padding_x = x_range * 0.05 if x_range > 0 else 1
    padding_y = y_range * 0.05 if y_range > 0 else 0.001
    ax.set_xlim(data["x"].min() - padding_x, data["x"].max() + padding_x)
    ax.set_ylim(data["y"].min() - padding_y, data["y"].max() + padding_y)
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig, ax
