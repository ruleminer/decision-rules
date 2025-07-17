try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.colors
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "To use visualization features, install all required packages: "
        "`pip install decision_rules[visualization]`"
    ) from e

from typing import Union, Optional, Any
import pandas as pd
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.ruleset import SurvivalRuleSet 
from decision_rules.survival.rule import SurvivalRule


def plot_kaplan_meier_curves(
    rules: Union[
        SurvivalRuleSet,
        SurvivalRule,
        list[SurvivalRule]
    ],
    X: Optional[pd.DataFrame] = None,
    y: Optional[Union[pd.Series, Any]] = None,
    survival_time_attribute: Optional[str] = None,
    show_default: bool = True,
    show_average: bool = True,
    show_uncovered: bool = False,
    ax: Optional[Axes] = None,
    show: bool = True,
    title: str = "Kaplan-Meier survival curves",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[
    tuple[Figure, Optional[Axes]],
    tuple[go.Figure, None]
]:
    """
    Plot Kaplan-Meier survival curves for a ruleset, list of rules, or single rule.

    Parameters
    ----------
    rules : SurvivalRuleSet, SurvivalRule, or list of SurvivalRule
        Rule set or rules to visualize.
    X : pandas.DataFrame, optional
        Feature data (required if show_uncovered is True).
    y : pandas.Series or array-like, optional
        Event indicator (required if show_uncovered is True).
    survival_time_attribute : str, optional
        Name of the column with survival time. Required if passing a list of rules or a single rule (not a RuleSet).
    show_default : bool, default=True
        Plot the global (all data) curve if available.
    show_average : bool, default=True
        Plot the average curve over all rules (if multiple).
    show_uncovered : bool, default=False
        Plot the curve for examples not covered by any rule (requires X and y and a ruleset).
    ax : matplotlib.axes.Axes or None, optional
        Axis for plotting (static). If None, a new axis is created.
    show : bool, default=True
        Whether to display the plot.
    title : str, optional
        Plot title.
    save_path : str or None, optional
        If provided, save the plot to this path.
    interactive : bool, default=False
        If True, use Plotly for an interactive plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
        The figure object.
    ax : matplotlib.axes.Axes or None
        The axis (matplotlib only).
    """
    # Normalize input
    if hasattr(rules, "rules"):
        ruleset = rules
        rules_list = ruleset.rules
    elif isinstance(rules, (list, tuple)):
        if survival_time_attribute is None:
            raise ValueError(
                "If passing a list of rules or a single rule, you must provide survival_time_attribute.")
        rules_list = list(rules)
        ruleset = SurvivalRuleSet(
            rules=rules_list, survival_time_attr=survival_time_attribute)
    else:
        if survival_time_attribute is None:
            raise ValueError(
                "If passing a single rule, you must provide survival_time_attribute.")
        rules_list = [rules]
        ruleset = SurvivalRuleSet(
            rules=rules_list, survival_time_attr=survival_time_attribute)

    uncovered_estimator = None
    not_rule_estimators = []
    not_rule_labels = []

    if ruleset is not None and X is not None and y is not None:
        X_np, y_np = ruleset._sanitize_dataset(X, y)
        coverage_matrix = ruleset.calculate_coverage_matrix(X_np)
        covered_mask = coverage_matrix.any(axis=1)
        if covered_mask.sum() > 0:
            times_covered = X.loc[covered_mask,
                                  ruleset.survival_time_attr_name].values
            events_covered = y.loc[covered_mask].values
            covered_estimator = KaplanMeierEstimator().fit(times_covered, events_covered)

    if show_uncovered and ruleset is not None and X is not None and y is not None:
        # Uncovered by ruleset (not covered by any rule)
        uncovered_mask = ~coverage_matrix.any(axis=1)
        if uncovered_mask.sum() > 0:
            times = X.loc[uncovered_mask,
                          ruleset.survival_time_attr_name].values
            events = y.loc[uncovered_mask].values
            uncovered_estimator = KaplanMeierEstimator().fit(times, events)
        # Not covered by each rule
        for i, rule in enumerate(ruleset.rules):
            not_rule_mask = ~coverage_matrix[:, i]
            if not_rule_mask.sum() > 0:
                times_not_rule = X.loc[not_rule_mask,
                                       ruleset.survival_time_attr_name].values
                events_not_rule = y.loc[not_rule_mask].values
                est = KaplanMeierEstimator().fit(times_not_rule, events_not_rule)
                not_rule_estimators.append(est)
                not_rule_labels.append(f"Not rule {i+1}")

    if interactive:
        fig = go.Figure()

        if show_default and ruleset is not None and hasattr(ruleset, "default_conclusion"):
            estimator = ruleset.default_conclusion.estimator
            fig.add_scatter(x=estimator.times, y=estimator.probabilities, mode="lines",
                            name="All data", line=dict(width=2, color="black"))

        if covered_estimator is not None:
            fig.add_scatter(
                x=covered_estimator.times,
                y=covered_estimator.probabilities,
                mode="lines",
                name="Covered by ruleset",
                line=dict(width=2, color="red")  # linia ciągła
            )

        rule_colors = plotly.colors.qualitative.Plotly

        for i, rule in enumerate(rules_list):
            est = rule.conclusion.estimator
            color = rule_colors[i % len(rule_colors)]
            fig.add_scatter(x=est.times, y=est.probabilities, mode="lines",
                            name=f"Rule {i+1}", line=dict(width=1.5, color=color))

        if show_average and len(rules_list) > 1:
            avg_est = KaplanMeierEstimator.average(
                [r.conclusion.estimator for r in rules_list])
            fig.add_scatter(x=avg_est.times, y=avg_est.probabilities, mode="lines",
                            name="Average (rules)", line=dict(width=2, color="blue", dash="dot"))

        for i, (est, label) in enumerate(zip(not_rule_estimators, not_rule_labels)):
            color = rule_colors[i % len(rule_colors)]
            fig.add_scatter(x=est.times, y=est.probabilities, mode="lines",
                            name=label, line=dict(width=1.5, color=color, dash="dot"))

        if uncovered_estimator is not None:
            fig.add_scatter(
                x=uncovered_estimator.times,
                y=uncovered_estimator.probabilities,
                mode="lines",
                name="Uncovered by ruleset",
                line=dict(width=2, color="red", dash="dot")  # linia przerywana
            )

        fig.update_layout(title=title, xaxis_title="Time",
                          yaxis_title="Survival probability")
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig, None

    fig, ax = plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax)

    if show_default and ruleset is not None and hasattr(ruleset, "default_conclusion"):
        estimator = ruleset.default_conclusion.estimator
        ax.step(estimator.times, estimator.probabilities,
                where="post", label="All data", lw=2, color="black")

    rule_colors = plotly.colors.qualitative.Plotly

    for i, rule in enumerate(rules_list):
        est = rule.conclusion.estimator
        label = f"Rule {i+1}: {str(rule)[:40]}..."
        ax.step(est.times, est.probabilities, where="post",
                label=label, lw=1.5, color=rule_colors[i])

    if show_average and len(rules_list) > 1:
        avg_est = KaplanMeierEstimator.average(
            [r.conclusion.estimator for r in rules_list])
        ax.step(avg_est.times, avg_est.probabilities, where="post",
                label="Average (rules)", lw=2, color="blue", linestyle=":")

    for i, (est, label) in enumerate(zip(not_rule_estimators, not_rule_labels)):
        color = rule_colors[i] if i < len(rule_colors) else None
        ax.step(est.times, est.probabilities, where="post",
                label=label, lw=1.5, linestyle="--", color=color)

    if covered_estimator is not None:
        ax.step(
            covered_estimator.times,
            covered_estimator.probabilities,
            where="post",
            label="Covered by ruleset",
            lw=2,
            color="red",
            linestyle="-",
        )

    if uncovered_estimator is not None:
        ax.step(
            uncovered_estimator.times,
            uncovered_estimator.probabilities,
            where="post",
            label="Uncovered by ruleset",
            lw=2,
            color="red",
            linestyle="--",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax
