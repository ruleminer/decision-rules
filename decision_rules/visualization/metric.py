import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

def plot_ruleset_metrics(model, X, y, x_metric, y_metric, interactive=True, show=True, save_path=None):
    """
    Create a 2D scatter plot for selected metrics for each rule.
    
    This function first computes the rule metrics by calling the model's
    calculate_rules_metrics(X, y) method and then generates either an interactive
    Plotly scatter plot or a static matplotlib scatter plot.
    
    Parameters
    ----------
    model : object
        A ruleset model which implements a method calculate_rules_metrics(X, y)
        and contains its rules in an attribute (e.g., model.rules).
    X : array-like
        The feature data.
    y : array-like
        The target data.
    x_metric : str
        The name of the metric to be plotted on the x-axis (e.g., "coverage").
    y_metric : str
        The name of the metric to be plotted on the y-axis (e.g., "precision").
    interactive : bool, optional (default=True)
        If True, create an interactive Plotly plot; otherwise, create a static matplotlib plot.
    show : bool, optional (default=True)
        If True, display the plot immediately.
    save_path : str, optional (default=None)
        If provided, save the interactive plot as an HTML file or the matplotlib plot as a PNG.
    
    Returns
    -------
    For interactive=True:
      fig : plotly.graph_objs._figure.Figure
          The interactive Plotly figure.
    For interactive=False:
      fig : matplotlib.figure.Figure
          The created matplotlib figure.
      ax : matplotlib.axes.Axes
          The axes object of the plot.
    """
    metrics = model.calculate_rules_metrics(X, y)
    rule_ids = list(metrics.keys())
    
    rule_texts = []
    for rid in rule_ids:
        rule_obj = next((r for r in model.rules if getattr(r, "uuid", None) == rid), None)
        if rule_obj is not None:
            rule_texts.append(str(rule_obj))
        else:
            rule_texts.append(rid)
    
    if interactive:
        # Prepare data for Plotly plot:
        x_values = [metrics[rid].get(x_metric, None) for rid in rule_ids]
        y_values = [metrics[rid].get(y_metric, None) for rid in rule_ids]
        
        # Create a DataFrame for plotting.
        df = pd.DataFrame({
            'Rule': rule_texts,
            x_metric: x_values,
            y_metric: y_values
        })
        
        # Create an interactive scatter plot using Plotly Express.
        fig = px.scatter(
            df, 
            x=x_metric, 
            y=y_metric, 
            hover_data=["Rule"],
            title=f"Relationship between {x_metric.capitalize()} and {y_metric.capitalize()}",
            labels={x_metric: x_metric.capitalize(), y_metric: y_metric.capitalize()}
        )
        
        # Update marker appearance using the desired color scheme.
        fig.update_traces(
            marker=dict(
                size=12,
                color="#C1572A",
                line=dict(width=1, color="#131720")
            ),
            selector=dict(mode="markers")
        )
        
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        
        if save_path is not None:
            fig.write_html(save_path)
        if show:
            fig.show()
        
        return fig
    else:
        # Prepare data for matplotlib plot:
        x_values = [metrics[rid].get(x_metric, np.nan) for rid in rule_ids]
        y_values = [metrics[rid].get(y_metric, np.nan) for rid in rule_ids]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(x_values, y_values, c='#435272', alpha=0.7, edgecolors='k')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax.set_xlabel(x_metric.capitalize())
        ax.set_ylabel(y_metric.capitalize())
        ax.set_title(f"Relationship between {x_metric.capitalize()} and {y_metric.capitalize()}")
        
        fig.tight_layout()
        
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        
        return fig, ax
