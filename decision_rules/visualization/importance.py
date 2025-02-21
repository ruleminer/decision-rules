import matplotlib.pyplot as plt

def plot_condition_importance(model, X, y, measure=None, show=True, save_path=None, max_conditions=10):
    """
    Plot condition importance for classification, regression, or survival rulesets.
    
    Parameters
    ----------
    model : object
        A ruleset model which implements a method calculate_condition_importances.
    X : array-like
        The feature data.
    y : array-like
        The target data.
    measure : callable or any, optional (default=None)
        The measure function or parameter used in calculating condition importances.
        If None, it will not be passed.
    show : bool, optional (default=True)
        If True, display the plot immediately.
    save_path : str, optional (default=None)
        If provided, save the figure to this file path (e.g., 'importance.png').
    max_conditions : int, optional (default=10)
        Maximum number of conditions to display per subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : list of matplotlib.axes.Axes
        A list of subplot axes.
    """
    if measure is not None:
        condition_importances = model.calculate_condition_importances(X, y, measure=measure)
    else:
        condition_importances = model.calculate_condition_importances(X, y)

    if isinstance(condition_importances, list):
        condition_importances = {"Condition Importance": condition_importances}
    
    keys = list(condition_importances.keys())
    n_keys = len(keys)
    
    # Create one subplot per key.
    fig, axes = plt.subplots(nrows=n_keys, ncols=1, figsize=(8, 4 * n_keys), squeeze=False)
    axes = axes.flatten()
    
    for idx, label in enumerate(keys):
        data = condition_importances[label]
        data = sorted(data, key=lambda x: x["importance"], reverse=True)
        data = data[:max_conditions]
        
        conditions = [item["condition"] for item in data]
        importances = [item["importance"] for item in data]
        
        ax = axes[idx]
        ax.barh(conditions, importances, color='#435272', alpha=0.8)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Condition")
        
        if n_keys == 1:
            ax.set_title("Condition Importance")
        else:
            ax.set_title(f"Condition importance for group: {label}")
        
        ax.invert_yaxis()  # Highest importance at the top
    
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    
    return fig, axes


def plot_attribute_importance(model, X, y, measure=None, show=True, save_path=None, max_attributes=10):
    """
    Plot attribute importance for classification, regression, or survival rulesets.
    
    Parameters
    ----------
    model : object
        A ruleset model that implements:
            - calculate_condition_importances(X, y, measure=...)
            - calculate_attribute_importances(condition_importances)
    X : array-like
        The feature data.
    y : array-like
        The target data.
    measure : callable or any, optional (default=None)
        The measure function or parameter used in calculating condition importances.
        If None, it will not be passed.
    show : bool, optional (default=True)
        If True, display the plot immediately.
    save_path : str, optional (default=None)
        If provided, save the figure as an image (e.g., 'attribute_importance.png').
    max_attributes : int, optional (default=10)
        Maximum number of attributes to display per plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    axes : list of matplotlib.axes.Axes
        A list of subplot axes.
    """
    # Compute condition importances.
    if measure is not None:
        condition_importances = model.calculate_condition_importances(X, y, measure=measure)
    else:
        condition_importances = model.calculate_condition_importances(X, y)
    
    # Compute attribute importances from condition importances.
    attribute_importances = model.calculate_attribute_importances(condition_importances)
    
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
    
    fig, axes = plt.subplots(nrows=n_keys, ncols=1, figsize=(8, 4 * n_keys), squeeze=False)
    axes = axes.flatten()
    
    for idx, label in enumerate(keys):
        data = attribute_importances[label]
        data = sorted(data, key=lambda x: x["importance"], reverse=True)
        data = data[:max_attributes]
        
        attributes = [item["attribute"] for item in data]
        importances = [item["importance"] for item in data]
        
        ax = axes[idx]
        ax.barh(attributes, importances, color='#D15E2E', alpha=0.8)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Attribute")
        
        if n_keys == 1:
            ax.set_title("Attribute Importance")
        else:
            ax.set_title(f"Attribute Importance for group: {label}")
        
        ax.invert_yaxis()  # Highest importance at the top
    
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    
    return fig, axes

