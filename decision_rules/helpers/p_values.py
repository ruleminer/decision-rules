def correct_p_values_fdr(p_values: list) -> list:
    """ Adjust p-values using the False Discovery Rate (FDR) method.

    Args:
        p_values (list): List of p-values to be adjusted.

    Returns:
        list: List of adjusted p-values, maintaining the original order.
    """
    N = len(p_values)
    p_values_with_index = [(pvalue, i) for i, pvalue in enumerate(p_values)]
    p_values_sorted = sorted(p_values_with_index)
    adjusted_p_values = [None] * N

    k = 1
    for p_value, index in p_values_sorted:
        adj_p = p_value * N / k
        adjusted_p_values[index] = adj_p
        k += 1
    return adjusted_p_values


def get_significant_fraction(p_values: list[float], significance_level: float) -> float:
    """Calculates the fraction of significant rules based on the p-values.

    Args:
        p_values (list[float]): List of p-values.
        significance_level (float): The significance level.

    Returns:
        float: The fraction of significant rules.
    """
    significant_rules_count = sum(p < significance_level for p in p_values)
    return significant_rules_count / len(p_values)
