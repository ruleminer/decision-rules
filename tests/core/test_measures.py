# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import types
import unittest

from decision_rules import measures
from decision_rules.core.coverage import Coverage


class TestMeasures(unittest.TestCase):

    def _get_all_measures(self) -> list[callable]:
        return list(
            filter(
                lambda member: isinstance(
                    member, types.FunctionType) and member.__name__ != "log_rank",
                [getattr(measures, name) for name in dir(measures)]
            )
        )

    def _test_all_measure(self, coverage_to_test: Coverage):
        failed_measures: list[tuple[str, str]] = []
        for measure in self._get_all_measures():
            try:
                measure(coverage_to_test)
            except Exception as error:
                failed_measures.append((measure.__name__, str(error)))
        if len(failed_measures) > 0:
            error_msg = 'Errors occured:\n'
            for measure_name, error in failed_measures:
                error_msg += f' -{measure_name}: {error}\n'
            self.fail(error_msg)

    def test_when_p_is_zero(self):
        self._test_all_measure(Coverage(p=0, n=4, P=10, N=10))

    def test_when_n_is_zero(self):
        self._test_all_measure(Coverage(p=4, n=0, P=10, N=10))

    def test_when_p_and_n_is_zero(self):
        self._test_all_measure(Coverage(p=0, n=0, P=10, N=10))

    def test_when_p_is_equal_P(self):
        self._test_all_measure(Coverage(p=10, n=4, P=10, N=10))

    def test_when_n_is_equal_N(self):
        self._test_all_measure(Coverage(p=4, n=10, P=10, N=10))

    def test_when_p_is_equal_P_and_n_is_equal_N(self):
        self._test_all_measure(Coverage(p=10, n=10, P=10, N=10))


if __name__ == '__main__':
    unittest.main()
