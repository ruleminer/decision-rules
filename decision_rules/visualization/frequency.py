try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "To use visualization features, install all required packages: "
        "`pip install decision_rules[visualization]`"
    ) from e
from typing import Optional, Union, Any
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule
from decision_rules.helpers.frequent import get_condition_frequent, get_attribute_frequent


def plot_condition_frequency(
    rules: Union[
        AbstractRuleSet,
        AbstractRule,
        list[AbstractRule]
    ],
    column_names: Optional[list] = None,
    max_conditions: int = 10,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Condition Frequency",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Visualizes the frequency of individual conditions in a rule set as a horizontal bar chart.

    Parameters
    ----------
    rules : AbstractRuleSet, AbstractRule, or list of AbstractRule
        Set of rules to visualize. Can be a RuleSet, a single Rule, or a list of Rule objects.
    column_names : list or None, optional
        List of attribute/column names. Required if `rules` is not a RuleSet.
    max_conditions : int, default=10
        Maximum number of most frequent conditions to display.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (only for matplotlib). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to display the plot immediately (calls plt.show() or fig.show()).
    title : str, default="Condition Frequency"
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
    # Normalize rules input
    if hasattr(rules, "rules") and hasattr(rules, "column_names"):
        rules_list = rules.rules
        column_names = rules.column_names
    elif isinstance(rules, (list, tuple)):
        rules_list = list(rules)
        if column_names is None:
            raise ValueError(
                "When passing a list of rules, column_names must also be provided.")
    condition_freq = get_condition_frequent(rules_list, column_names)

    sorted_conditions = sorted(condition_freq.items(
    ), key=lambda x: x[1], reverse=True)[:max_conditions]
    conditions = [item[0] for item in sorted_conditions]
    frequencies = [item[1] for item in sorted_conditions]

    if interactive:
        fig = go.Figure(go.Bar(
            x=frequencies,
            y=conditions,
            orientation='h',
            marker_color='#435272',
            opacity=0.85
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Condition",
            yaxis=dict(
                tickmode='linear',
                dtick=1,
                tick0=0,
                tickformat=',d'
            ),
            bargap=0.25
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)
    ax.barh(conditions, frequencies, color='#435272', alpha=0.8)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Condition")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


def plot_attribute_frequency(
    rules: Union[
        AbstractRuleSet,
        AbstractRule,
        list[AbstractRule]
    ],
    column_names: Optional[list] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Attribute Frequency",
    max_attributes: int = 10,
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Visualizes the frequency of attributes in a rule set as a horizontal bar chart.

    Parameters
    ----------
    ruleset : AbstractRuleSet
        Rule set object supporting the get_attribute_frequent() or get_attribute_occurrences() method.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw the plot on (only for matplotlib). If None, a new figure and axis are created.
    show : bool, default=True
        Whether to display the plot immediately (calls plt.show() or fig.show()).
    title : str, default="Attribute Frequency"
        Title of the plot.
    max_attributes : int, default=10
        Maximum number of most frequent attributes to display.
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
    # Normalize rules input
    if hasattr(rules, "rules") and hasattr(rules, "column_names"):
        rules_list = rules.rules
        column_names = rules.column_names
    elif isinstance(rules, (list, tuple)):
        rules_list = list(rules)
        if column_names is None:
            raise ValueError(
                "When passing a list of rules, column_names must also be provided.")

    attribute_freq = get_attribute_frequent(rules_list, column_names)

    sorted_attributes = sorted(attribute_freq.items(
    ), key=lambda x: x[1], reverse=True)[:max_attributes]
    attributes = [item[0] for item in sorted_attributes]
    frequencies = [item[1] for item in sorted_attributes]

    if interactive:
        fig = go.Figure(go.Bar(
            x=frequencies,
            y=attributes,
            orientation='h',
            marker_color='#D15E2E',
            opacity=0.85
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Attribute",
            yaxis=dict(
                tickmode='linear',
                dtick=1,
                tick0=0,
                tickformat=',d'
            ),
            bargap=0.25
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)
    ax.barh(attributes, frequencies, color='#D15E2E', alpha=0.8)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Attribute")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax
