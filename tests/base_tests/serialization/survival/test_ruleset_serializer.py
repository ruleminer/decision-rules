# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

from decision_rules.conditions import AttributesCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.coverage import Coverage
from decision_rules.serialization import JSONSerializer
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet


class TestSurvivalRuleSetSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        rule1 = SurvivalRule(
            CompoundCondition(
                subconditions=[
                    AttributesCondition(
                        column_left=2, column_right=3, operator='>'
                    ),
                    ElementaryCondition(
                        column_index=2, left=-1, right=2.0, left_closed=True, right_closed=False
                    ),
                    NominalCondition(
                        column_index=2,
                        value='value',
                    )
                ],
                logic_operator=LogicOperators.ALTERNATIVE
            ),
            conclusion=SurvivalConclusion(1.0, column_name='label'),
            column_names=['col_1', 'col_2', 'col_3', 'col_4', 'survival_time'],
            survival_time_attr='survival_time'
        )
        rule1.coverage = Coverage(p=10, n=2, P=12, N=20)
        rule1.conclusion.median_survival_time_ci_lower = 1.0
        rule1.conclusion.median_survival_time_ci_upper = 3.0
        rule2 = SurvivalRule(
            CompoundCondition(
                subconditions=[
                    AttributesCondition(
                        column_left=1, column_right=3, operator='='
                    ),
                    ElementaryCondition(
                        column_index=2,
                        left=float('-inf'),
                        right=3.0,
                        left_closed=False,
                        right_closed=False
                    ),
                ],
                logic_operator=LogicOperators.CONJUNCTION
            ),
            conclusion=SurvivalConclusion(1.0, column_name='label'),
            column_names=['col_1', 'col_2', 'col_3', 'col_4', 'survival_time'],
            survival_time_attr='survival_time'
        )
        rule2.coverage = Coverage(p=19, n=1, P=20, N=12)
        rule2.conclusion.median_survival_time_ci_lower = 1.0
        rule2.conclusion.median_survival_time_ci_upper = 3.0
        ruleset = SurvivalRuleSet(
            rules=[rule1, rule2], survival_time_attr='survival_time')
        conclusion_estimator_dict = {'times': [1.1, 2.1, 3.0, 10, 22.2],
                                     'events_count': [1.0, 0, 1.0, 1.0, 0],
                                     'censored_count': [0.0, 1.0, 2.0, 0.0, 5.0],
                                     'at_risk_count': [75.0, 21.0, 5.0, 1.0, 0.0],
                                     'probabilities': [0.98, 0.92, 0.76, 0.52, 0.31]}
        ruleset.default_conclusion = SurvivalConclusion(
            value=None,
            column_name=ruleset.rules[0].conclusion.column_name
        )
        ruleset.default_conclusion.estimator = KaplanMeierEstimator().update(
            conclusion_estimator_dict)
        ruleset.column_names = ['col_1', 'col_2',
                                'col_3', 'col_4', 'survival_time']

        serialized_ruleset = JSONSerializer.serialize(ruleset)
        deserializer_ruleset = JSONSerializer.deserialize(
            serialized_ruleset, SurvivalRuleSet
        )

        self.assertEqual(
            ruleset, deserializer_ruleset,
            'Serializing and deserializing should lead to the the same object'
        )


if __name__ == '__main__':
    unittest.main()
