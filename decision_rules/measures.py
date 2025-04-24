"""
Contains functions for calculating different qualities measures.
They can be used for calculating rule qualities and voting weights
"""
import math
import sys

from decision_rules.core.coverage import Coverage as Cov


def accuracy(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return float(c.p - c.n)


def kappa(c: Cov) -> float:  # pylint: disable=invalid-name,missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    return (
        ((c.P + c.N) * (c.p / (c.p + c.n)) - c.P) /
        ((c.P + c.N) / 2 * ((c.p + c.n + c.P) / (c.p + c.n)) - c.P)
    )


def c1(c: Cov) -> float:  # pylint: disable=invalid-name,missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    cohen: float = kappa(c)
    return ((c.N * c.p - c.P * c.n) / (c.N * (c.p + c.n))) * ((2.0 + cohen) / 3.0)


def c2(c: Cov) -> float:  # pylint: disable=invalid-name,missing-function-docstring
    if c.p == 0:
        return 0.0
    if c.p + c.n == 0:
        return 0.0
    if c.N == 0:
        return 0.0

    return (
        (((c.P + c.N) * c.p / (c.p + c.n) - c.P) / c.N) *
        ((1 + c.p / c.P) / 2)
    )


def c_foil(c: Cov) -> float:  # pylint: disable=invalid-name,missing-function-docstring
    if c.p == 0:
        return 0.0
    return c.p * (math.log2(c.p / (c.p + c.n)) - math.log2(c.P / (c.P + c.N)))


def coverage(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return c.p / c.P


def cn2_significnce(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 or c.n == 0:
        return 0.0
    return 2 * (
        c.p * math.log(c.p / ((c.p + c.n) * c.P / (c.P + c.N))) +
        c.n * math.log(c.n / ((c.p + c.n) * c.N / (c.P + c.N)))
    )


def full_coverage(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (c.p + c.n) / (c.P + c.N)


def laplace(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (c.p + 1) / (c.p + c.n + 2)


def weighted_laplace(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (c.p + 1) * (c.P + c.N) / ((c.p + c.n + 2) * c.P)


def specificity(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (c.N - c.n) / c.N


def sensitivity(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return c.p / c.P


def lift(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if (c.p + c.n) == 0:
        return 0.0
    return c.p * (c.P + c.N) / ((c.p + c.n) * c.P)


def precision(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if (c.p + c.n) == 0:
        return 0
    return c.p / (c.p + c.n)


def correlation(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    denominator: float = math.sqrt(
        c.P * c.N * (c.p + c.n) * (c.P - c.p + c.N - c.n)
    )
    if denominator == 0.0:
        return 0.0
    return (c.p * c.N - c.P * c.n) / denominator


def rss(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return c.p / c.P - c.n / c.N


def odds_ratio(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (
        (c.p * (c.N - c.n)) /
        (c.n * (c.P - c.p)+1)
    )


def f_bayesian_confirmation(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    return (c.p * c.N - c.n * c.P) / (c.p * c.N + c.n * c.P)


def f_measure(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    beta_2: float = 2 * 2
    if c.p == 0:
        return 0.0
    return (
        (beta_2 + 1) * (c.p / (c.p + c.n)) *
        (c.p / c.P) / (beta_2 * (c.p / (c.p + c.n) + c.p / c.P))
    )


def geo_rss(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return math.sqrt(c.p / c.P * (1 - c.n / c.N))


def g_measure(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    g: float = 2
    return c.p / (c.p + c.n + g)


def information_gain(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    consequent: float = c.P
    not_consequent: float = c. N

    antecedent: float = c.p + c.n
    not_antecedent: float = c.P + c.N - c.p - c.n

    antecedent_and_consequent: float = c.p
    antecedent_but_not_consequent = antecedent - antecedent_and_consequent

    not_antecedent_and_not_consequent: float = not_consequent - \
        antecedent_but_not_consequent
    not_antecedent_but_consequent: float = not_antecedent - \
        not_antecedent_and_not_consequent

    v: float = consequent + not_consequent

    a: float = consequent / v
    b: float = not_consequent / v

    info_all_examples: float
    if b > 0:
        info_all_examples = -(a * math.log2(a) + b * math.log2(b))
    else:
        info_all_examples = -(a * math.log2(a))

    info_matched_examples: float = 0.0
    if (
        antecedent_and_consequent != 0 and
        antecedent_but_not_consequent != 0
    ):  # if rule is not accurate
        a = antecedent_and_consequent / antecedent
        b = antecedent_but_not_consequent / antecedent
        info_matched_examples = -(a * math.log2(a) + b * math.log2(b))

    info_not_matched_examples: float = 0.0
    if not_antecedent_but_consequent != 0 and not_antecedent_and_not_consequent != 0:
        a = not_antecedent_but_consequent / not_antecedent
        b = not_antecedent_and_not_consequent / not_antecedent
        info_not_matched_examples = -(a * math.log2(a) + b * math.log2(b))

    c: float = antecedent / v
    info_rule: float = c * info_matched_examples + \
        (1 - c) * info_not_matched_examples

    info: float = info_all_examples - info_rule

    if (
        antecedent_but_not_consequent > 0 and
        antecedent_and_consequent / antecedent_but_not_consequent < consequent / not_consequent
    ):
        # this makes measure monotone
        info = -info
    return info


def j_measure(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0:
        a: float = 0.0
    else:
        a: float = c.p * math.log(c.p * (c.P + c.N) / ((c.p + c.n) * c.P))
    if c.n == 0:
        b: float = 0.0
    else:
        b: float = c.n * math.log(c.n * (c.P + c.N) / ((c.p + c.n) * c.N))

    return (1.0 / (c.P + c.N)) * (a + b)


def klosgen(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    omega: float = 1.0
    return math.pow((c.p + c.n) / (c.P + c.N), omega) * (c.p / (c.p + c.n) - c.P / (c.P + c.N))


def logical_sufficiency(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.n == 0 or c.P == 0:
        return sys.float_info.max
    return c.p + c.N / (c.n * c.P)


def m_estimate(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    m: float = 2.0
    return (c.p + m * c.P / (c.P + c.N)) / (c.p + c.n + m)


def mutual_support(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return c.p / (c.n + c.P)


def novelty(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return c.p / (c.P + c.N) - (c.P * (c.p + c.n) / ((c.P + c.N) * (c.P + c.N)))


def one_way_support(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0.0:
        return 0.0
    return c.p / (c.p + c.n) * math.log(c.p * (c.P + c.N) / ((c.p + c.n) * c.P))


def pawlak_dependency_factor(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    return (c.p * (c.P + c.N) - c.P * (c.p + c.n)) / (c.p * (c.P + c.N) + c.P * (c.p + c.n))


def q2(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (c.p / c.P - c.n / c.N) * (1 - c.n / c.N)


def relative_risk(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    return (c.p / (c.p + c.n)) * ((c.P + c.N - c.p - c.n) / (c.P - c.p + 1))


def ripper(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    return (c.p - c.n) / (c.p + c.n)


def rule_interest(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (c.p * (c.P + c.N) - (c.p + c.n) * c.P) / (c.P + c.N)


def s_bayesian(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == c.P and c.n == c.N:
        return 0.0
    if c.p == 0 and c.n == 0:
        return - (c.P - c.p) / (c.P - c.p + c.N - c.n)
    return c.p / (c.p + c.n) - (c.P - c.p) / (c.P - c.p + c.N - c.n)


def two_way_support(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0.0:
        return 0.0
    return (c.p / (c.P + c.N)) * math.log(c.p * (c.P + c.N) / ((c.p + c.n) * c.P))


def weighted_relative_accuracy(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    return (c.p + c.n) / (c.P + c.N) * (c.p / (c.p + c.n) - c.P / (c.P + c.N))


def yails(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    if c.p == 0 and c.n == 0:
        return 0.0
    prec = precision(c)
    w1: float = 0.5 + 0.25 * prec
    w2: float = 0.5 - 0.25 * prec
    return w1 * c.p / (c.p + c.n) + w2 * (c.p / c.P)


def confidence(c: Cov) -> float:  # pylint: disable=missing-function-docstring
    return (c.P + c.N) / (c.P + c.N + c.n + c.p)

def lord(c: Cov, m: float = 0.1) -> float:
    """
    m‐estimate heuristic value used in LORD algorithm.

    Based on the m‐estimate from Cestnik (1990) and employed in CN2 (Clark & Boswell, 1991)
    and Ripper (Cohen, 1995). In LORD, this measure guides rule growth and pruning
    to find locally optimal rules (see “Efficient learning of large sets of locally optimal
    classification rules”, Machine Learning 112:571–610, 2023;
    https://doi.org/10.1007/s10994-022-06290-w).

    Formula:
        h_m = (p + m * (P / (P + N))) / (p + n + m)

    where:
      p = number of true positives (c.p)
      n = number of false positives (c.n)
      P = total number of positives (c.P)
      N = total number of negatives (c.N)
      m = smoothing parameter (default 0.1)
    """
    if c.p + c.n + m == 0:
        return 0.0
    return (c.p + m * (c.P / (c.P + c.N))) / (c.p + c.n + m)
