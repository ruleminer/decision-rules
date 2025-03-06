from pyvis.network import Network
import webbrowser
import os
import numpy as np
import pandas as pd

def visualize_rules_graph(rules, X, y, problem_type="classification"):
    """
    Visualizes an interactive graph of rules as a sequence:
      rule -> conditions (chain) -> conclusion.
    For classification, edges show [positive, negative] counts.
    For regression, edges show "mean ± std".
    For survival, edges show median survival time.
    The graph is generated using the PyVis library.
    """
    # Convert X and y to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    # Default styling for nodes and colors
    default_node_color = "#5F73A1"
    conclusion_color   = "#C1572A"
    rule_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    net = Network(height="750px", width="100%", directed=True)
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
        """
        Computes the label for an edge based on the current rule and its accumulated conditions.
        """
        temp_premise = rule.premise.__class__(subconditions=accumulated_conditions)
        temp_rule = rule.__class__(premise=temp_premise, conclusion=rule.conclusion,
                                   column_names=rule.column_names)
        mask = temp_rule.premise.covered_mask(X)
        if np.count_nonzero(mask) == 0:
            return "N/A" if problem_type in ["regression", "survival"] else "[N/A]"

        if problem_type == "regression":
            mean_val = np.mean(y[mask])
            std_val = np.std(y[mask])
            return f"{mean_val:.2f} ± {std_val:.2f}"
        elif problem_type == "survival":
            temp_rule.set_survival_time_attr("survival_time")
            _ = temp_rule.calculate_coverage(X, y)
            return f"{temp_rule.conclusion.value:.2f}" if temp_rule.conclusion.value is not None else "N/A"
        else:  # classification
            coverage = temp_rule.calculate_coverage(X, y)
            return f"[{coverage.p}, {coverage.n}]"

    # Dictionaries to track nodes that have already been added
    condition_nodes = {}   # Maps condition string -> node id
    conclusion_nodes = {}  # Maps conclusion string -> node id

    for i, rule in enumerate(rules):
        concl_str = str(rule.conclusion)
        rule_color = rule_palette[i % len(rule_palette)]
        rule_label = getattr(rule, "name", f"Rule_{i}")
        rule_id = f"rule_{i}"
        net.add_node(rule_id, label=rule_label, color=rule_color, shape="box")

        # Obtain conditions from the rule premise
        if hasattr(rule.premise, "subconditions") and rule.premise.subconditions:
            conditions = rule.premise.subconditions
        else:
            conditions = [rule.premise]

        # Compute baseline label with no conditions
        baseline_label = compute_label(rule, [])

        # Add edge from the rule node to the first condition if available
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

        # Add edges for subsequent conditions
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

        # Add edge from the last condition to the conclusion
        if concl_str not in conclusion_nodes:
            concl_id = f"concl_{len(conclusion_nodes)}"
            conclusion_nodes[concl_str] = concl_id
            net.add_node(concl_id, label=concl_str, color=conclusion_color, shape="ellipse")
        else:
            concl_id = conclusion_nodes[concl_str]
        final_edge_label = compute_label(rule, accumulated_conditions)
        net.add_edge(current_node, concl_id, color=rule_color, width=3, length=300,
                     label=final_edge_label)

    # Generate and open the HTML visualization
    html = net.generate_html(notebook=False)
    output_file = ("interactive_regression_rules_graph.html" if problem_type=="regression"
                   else "interactive_survival_rules_graph.html" if problem_type=="survival"
                   else "interactive_classification_rules_graph.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    webbrowser.open("file://" + os.path.realpath(output_file))