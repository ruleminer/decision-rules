import numpy as np
import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.measures import *
from decision_rules.problem import ProblemTypes
from pydantic import BaseModel


class Histograms(BaseModel):
    min: int
    max: int
    bin_edges: list[float]
    histograms: dict[str, list[int]]


def get_histograms(
        model: AbstractRuleSet, dataset: pd.DataFrame,
        problem_type: ProblemTypes, bins: int, for_rules: list[str] = ()
) -> Histograms:
    # check if problem type is supported
    if problem_type in [ProblemTypes.CLASSIFICATION, ProblemTypes.SURVIVAL]:
        raise NotImplementedError(
            f'Histogram for {problem_type} type of problem is not supported yet'
        )
    # get rules to process
    if not for_rules:
        rules_to_process = model.rules
    else:
        rules_to_process = [
            rule for rule in model.rules if rule.uuid in for_rules]
    # get X and y
    X, y = model.split_dataset(dataset)
    # calculate
    data = {}
    max_histogram = 0
    min_histogram = float('inf')
    X = X[model.column_names].to_numpy()
    y_min = y.min()
    y_max = y.max()
    for rule in rules_to_process:
        rule_uuid = rule.uuid
        covered_y = y[rule.premise.covered_mask(X)]
        histogram, bin_edges = np.histogram(
            covered_y, bins=bins, range=(y_min, y_max))

        data[rule_uuid] = histogram.tolist()
        if max_histogram < histogram.max():
            max_histogram = histogram.max()
        if min_histogram > histogram.min():
            min_histogram = histogram.min()

    return Histograms(
        min=min_histogram,
        max=max_histogram,
        histograms=data,
        bin_edges=bin_edges.tolist()
    )
