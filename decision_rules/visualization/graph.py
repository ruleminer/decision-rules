from pyvis.network import Network
import webbrowser
import os

def visualize_rules_graph(rules):
    """
    Visualizes an interactive graph of rules as a sequence:
      rule -> conditions (chain) -> conclusion.
      
    Parameters:
      rules (list): List of rules (e.g. ClassificationRule),
                    where each rule has:
                      - premise: an object inheriting from AbstractCondition
                      - conclusion: a conclusion object (converted to string)
                      - column_names: list[str]
                      - optionally: name (str)
    """
    default_node_color = "#5F73A1"
    conclusion_color   = "#C1572A"  
    # Palette of unique colors for the rule nodes (start of each chain)
    rule_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                    
    filtered_rules = rules

    net = Network(height="750px", width="100%", directed=True)

    # Settings for hierarchical layout (LR - Left to Right), physics, edge appearance and interaction
    net.set_options('''
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "nodeSpacing": 200,
          "levelSeparation": 150
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -600,
          "centralGravity": 0.1,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.8,
          "avoidOverlap": 0.3
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

    condition_nodes = {}  # Map: condition (string) -> node id
    conclusion_nodes = {} # Map: conclusion (string) -> node id

    for i, rule in enumerate(filtered_rules):
        rule_color = rule_palette[i % len(rule_palette)]
        rule_label = getattr(rule, "name", f"Rule_{i}")
        rule_id = f"rule_{i}"
        net.add_node(rule_id, label=rule_label, color=rule_color, shape="box")
        
        conditions = []
        if hasattr(rule.premise, "subconditions") and rule.premise.subconditions:
            for sub in rule.premise.subconditions:
                conditions.append(sub.to_string(rule.column_names))
        else:
            cond_str = rule.premise.to_string(rule.column_names)
            if " AND " in cond_str:
                parts = [part.strip() for part in cond_str.split(" AND ") if part.strip()]
                conditions.extend(parts)
            else:
                conditions.append(cond_str)
        
        current_node = rule_id
        for cond in conditions:
            if cond not in condition_nodes:
                cond_id = f"condition_{len(condition_nodes)}"
                condition_nodes[cond] = cond_id
                net.add_node(cond_id, label=cond, color=default_node_color)
            else:
                cond_id = condition_nodes[cond]
            net.add_edge(current_node, cond_id, color=rule_color, width=3)
            current_node = cond_id
        
        concl_str = str(rule.conclusion)
        if concl_str not in conclusion_nodes:
            concl_id = f"concl_{len(conclusion_nodes)}"
            conclusion_nodes[concl_str] = concl_id
            net.add_node(concl_id, label=concl_str, color=conclusion_color, shape="ellipse")
        else:
            concl_id = conclusion_nodes[concl_str]
            
        net.add_edge(current_node, concl_id, color=rule_color, width=3)

    html = net.generate_html(notebook=False)
    output_file = "interactive_classification_rules_graph.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    webbrowser.open("file://" + os.path.realpath(output_file))