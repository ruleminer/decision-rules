import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from typing import Optional, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import plotly.graph_objects as go
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule


def extract_rule_profile(rule, column_names):
    """
    Extracts the attribute profile of a rule (list of attributes in order of occurrence).
    """
    profile = []

    def traverse(condition):
        if hasattr(condition, "subconditions") and condition.subconditions:
            for sub in condition.subconditions:
                traverse(sub)
        else:
            if hasattr(condition, "column_left") and hasattr(condition, "column_right"):
                profile.append(column_names[condition.column_left])
                profile.append(column_names[condition.column_right])
            elif hasattr(condition, "column_index"):
                profile.append(column_names[condition.column_index])

    traverse(rule.premise)
    return profile


def attribute_lexicographical_sort_key(attr, attr_position_counts, max_length):
    """Sort key for lexicographical sorting of attributes by their position counts."""
    return tuple([-attr_position_counts[attr][pos] for pos in range(max_length)])


def plot_rules_profile(
    rules: Union[
        AbstractRuleSet,
        AbstractRule,
        list[AbstractRule]
    ],
    column_names: Optional[list] = None,
    rule_indices: Optional[list] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Parallel Plot of Rule Profiles",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Plot parallel coordinates of attribute order in rules (rule profiles).

    Parameters
    ----------
    rules : AbstractRuleSet, list of AbstractRule, or AbstractRule
        RuleSet object (must have `.rules` and `.column_names`), list of rules, or single rule.
    column_names : list or None, optional
        List of attribute/column names. Required if `rules` is not a RuleSet.
    rule_indices : list or None, optional
        Indices of rules to plot. If None, plot all.
    ax : matplotlib.axes.Axes or None, optional
        Axis for static plotting. If None, create new axis.
    show : bool, default=True
        Whether to display the plot immediately.
    title : str, optional
        Plot title.
    save_path : str or None, optional
        If provided, save the plot to this path.
    interactive : bool, default=False
        If True, use Plotly for interactive plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The figure object.
    ax : matplotlib.axes.Axes or None
        Axis object (matplotlib only).
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
    else:
        rules_list = [rules]
        if column_names is None:
            raise ValueError(
                "When passing a single rule, column_names must be provided.")

    if rule_indices is None:
        rule_indices = range(len(rules_list))

    # Build profiles
    profiles = [extract_rule_profile(
        rules_list[i], column_names) for i in rule_indices]
    profiles = [p for p in profiles if len(p) > 0]
    if not profiles:
        raise ValueError("No attributes to display in the rule profile.")

    max_length = max(len(profile) for profile in profiles)
    attr_position_counts = defaultdict(lambda: [0] * max_length)
    for profile in profiles:
        for pos, attr in enumerate(profile):
            attr_position_counts[attr][pos] += 1
    all_attrs = list(attr_position_counts.keys())
    sorted_attributes = sorted(
        all_attrs,
        key=lambda attr: attribute_lexicographical_sort_key(
            attr, attr_position_counts, max_length),
    )

    # Interactive plot (Plotly)
    if interactive:
        fig = go.Figure()
        for idx, profile in zip(rule_indices, profiles):
            positions = list(range(1, len(profile) + 1))
            y = [sorted_attributes.index(attr) for attr in profile]
            if len(profile) == 1:
                fig.add_trace(go.Scatter(
                    x=positions,
                    y=y,
                    mode="markers",
                    name=f"Rule {idx}",
                    text=[f"Rule {idx}, Attr: {profile[0]}"],
                    hoverinfo="text"
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=positions,
                    y=y,
                    mode="lines",
                    name=f"Rule {idx}",
                    text=[f"Rule {idx}, Attr: {attr}" for attr in profile],
                    hoverinfo="text"
                ))
        fig.update_layout(
            title=title,
            xaxis_title="Position in rule",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(sorted_attributes))),
                ticktext=sorted_attributes
            ),
            yaxis_title="Attribute",
            height=400 + 15 * len(sorted_attributes),
            legend_title="Rule",
            xaxis=dict(
                tickmode='linear',
                dtick=1,
                tick0=1,
                tickformat=',d'
            ),
        )
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    # Static plot (matplotlib)
    fig, ax = plt.subplots(figsize=(
        10, 2 + 0.5 * len(sorted_attributes))) if ax is None else (ax.figure, ax)
    for idx, profile in zip(rule_indices, profiles):
        L = len(profile)
        positions = list(range(1, L + 1))
        y = [sorted_attributes.index(attr) for attr in profile]
        if L == 1:
            ax.plot(positions, y, linestyle="None",
                    marker="o", label=f"Rule {idx}")
        else:
            ax.plot(positions, y, linestyle="-", label=f"Rule {idx}")

    ax.set_xlabel("Position in rule")
    ax.set_ylabel("Attribute")
    ax.set_title(title)
    ax.set_xticks(range(1, max_length + 1))
    ax.set_yticks(range(len(sorted_attributes)))
    ax.set_yticklabels(sorted_attributes)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax
