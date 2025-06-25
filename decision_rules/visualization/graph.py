import numpy as np
import pandas as pd
import os
import webbrowser
import matplotlib
import matplotlib.pyplot as plt
from pyvis.network import Network
from typing import Optional, Union, Any
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule


def visualize_rules_graph(
    rules: Union[
        AbstractRuleSet,
        AbstractRule,
        list[AbstractRule]
    ],
    X: pd.DataFrame,
    y: Union[pd.Series, Any],
    problem_type: str = "classification",
    survival_time_attr: str = None,
    height: str = "750px",
    width: str = "100%",
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Visualizes an interactive graph of rules as a sequence: rule → conditions (chain) → conclusion using PyVis.

    Parameters
    ----------
    rules : AbstractRuleSet, AbstractRule, or list of AbstractRule
        Set of rules to visualize. Can be a RuleSet, a single Rule, or a list of Rule objects.
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix (samples x features) for evaluating rule coverage.
    y : pandas.Series, numpy.ndarray, or array-like
        Target vector (ground truth labels or values).
    problem_type : str, default="classification"
        Type of problem: "classification", "regression", or "survival".
    survival_time_attr : str or None, optional
        Name of the survival time attribute (only for survival analysis).
    height : str, default="750px"
        Height of the PyVis network visualization.
    width : str, default="100%"
        Width of the PyVis network visualization.
    show : bool, default=True
        Whether to open the generated visualization in a browser.
    save_path : str or None, optional
        If provided, saves the HTML visualization to this path. Otherwise, uses a default file name.

    Returns
    -------
    output_file : str
        Path to the saved HTML file with the interactive visualization.
    """


    # Normalize rules input
    if hasattr(rules, "rules"):
        rules = rules.rules
    elif isinstance(rules, (list, tuple)):
        rules = list(rules)
    else:
        rules = [rules]

    if isinstance(X, pd.DataFrame):
        X_np = X.to_numpy()
    else:
        X_np = np.asarray(X)
    if isinstance(y, pd.Series):
        y_np = y.to_numpy()
    else:
        y_np = np.asarray(y)

    default_node_color = "#5F73A1"
    conclusion_color = "#C1572A"
    rule_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    if problem_type == "classification":
        class_labels = sorted({str(rule.conclusion) for rule in rules})
        n_classes = len(class_labels)
        cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
        conclusion_palette = [matplotlib.colors.to_hex(
            cmap(i % cmap.N)) for i in range(n_classes)]
        class2color = {cl: conclusion_palette[i]
                       for i, cl in enumerate(class_labels)}

    net = Network(height=height, width=width, directed=True)
    net.set_options('''
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 200,
          "nodeSpacing": 300
        }
      },
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 150,
          "nodeDistance": 150
        },
        "minVelocity": 0.75
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "curvedCW",
          "roundness": 0.2
        },
        "color": {"inherit": false},
        "width": 3
      },
      "interaction": {
        "dragNodes": true
      }
    }
    ''')

    def compute_label(rule, accumulated_conditions):
        temp_premise = rule.premise.__class__(
            subconditions=accumulated_conditions)
        temp_rule = rule.__class__(premise=temp_premise, conclusion=rule.conclusion,
                                   column_names=rule.column_names)
        mask = temp_rule.premise.covered_mask(X_np)
        if np.count_nonzero(mask) == 0:
            return "N/A" if problem_type in ["regression", "survival"] else "[N/A]"

        if problem_type == "regression":
            mean_val = np.mean(y_np[mask])
            std_val = np.std(y_np[mask])
            return f"{mean_val:.2f} ± {std_val:.2f}"
        elif problem_type == "survival":
            survival_time = survival_time_attr or "survival_time"
            temp_rule.set_survival_time_attr(survival_time)
            _ = temp_rule.calculate_coverage(X_np, y_np)
            return f"{temp_rule.conclusion.value:.2f}" if temp_rule.conclusion.value is not None else "N/A"
        else:
            coverage = temp_rule.calculate_coverage(X_np, y_np)
            return f"[{coverage.p}, {coverage.n}]"

    condition_nodes = {}
    conclusion_nodes = {}

    for i, rule in enumerate(rules):
        concl_str = str(rule.conclusion)
        rule_color = rule_palette[i % len(rule_palette)]
        rule_label = getattr(rule, "name", f"Rule_{i}")
        rule_id = f"rule_{i}"
        net.add_node(rule_id, label=rule_label, color=rule_color, shape="box")

        if hasattr(rule.premise, "subconditions") and rule.premise.subconditions:
            conditions = rule.premise.subconditions
        else:
            conditions = [rule.premise]

        baseline_label = compute_label(rule, [])

        if conditions:
            first_cond = conditions[0]
            cond_str = first_cond.to_string(rule.column_names)
            if cond_str not in condition_nodes:
                cond_id = f"condition_{len(condition_nodes)}"
                condition_nodes[cond_str] = cond_id
                net.add_node(cond_id, label=cond_str, color=default_node_color)
            else:
                cond_id = condition_nodes[cond_str]
            net.add_edge(rule_id, cond_id, color=rule_color, width=3,
                         label=baseline_label, length=300)
            current_node = cond_id
            accumulated_conditions = [first_cond]
        else:
            current_node = rule_id
            accumulated_conditions = []

        for cond in conditions[1:]:
            cond_str = cond.to_string(rule.column_names)
            if cond_str not in condition_nodes:
                cond_id = f"condition_{len(condition_nodes)}"
                condition_nodes[cond_str] = cond_id
                net.add_node(cond_id, label=cond_str, color=default_node_color)
            else:
                cond_id = condition_nodes[cond_str]
            edge_label = compute_label(rule, accumulated_conditions)
            net.add_edge(current_node, cond_id, color=rule_color, width=3,
                         label=edge_label, length=300)
            accumulated_conditions.append(cond)
            current_node = cond_id

        if concl_str not in conclusion_nodes:
            concl_id = f"concl_{len(conclusion_nodes)}"
            conclusion_nodes[concl_str] = concl_id
            if problem_type == "classification":
                concl_color = class2color[concl_str]
            else:
                concl_color = conclusion_color
            net.add_node(concl_id, label=concl_str,
                         color=concl_color, shape="ellipse")
        else:
            concl_id = conclusion_nodes[concl_str]
        final_edge_label = compute_label(rule, accumulated_conditions)
        net.add_edge(current_node, concl_id, color=rule_color, width=3, length=300,
                     label=final_edge_label)

    if save_path:
        output_file = save_path
    else:
        if problem_type == "regression":
            output_file = "interactive_regression_rules_graph.html"
        elif problem_type == "survival":
            output_file = "interactive_survival_rules_graph.html"
        else:
            output_file = "interactive_classification_rules_graph.html"

    html = net.generate_html()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    if show:
        webbrowser.open("file://" + os.path.realpath(output_file))
    return output_file
