# Decision Rules

Package implementing decision rules.
The package allows user to define and process decision rules as Python objects
and perform operations on datasets to which these rules apply.

Three types of problems are supported:

- classification
- regression
- survival

Functionalities includes, but is not limited to:

- serialization and deserialization
- prediction
- summary statistics
- comparison between rules (semantic and syntactic)

## Installation
Base package can be installed from [PyPi](https://pypi.org/project/decision-rules/):
```
pip install decision-rules
```

Besides the base package, additional dependencies are required to use the `ruleset_factories` module. To install these extras, run:
```
pip install decision-rules[ruleset_factories]
```

You can also just clone the repository and install the pacakage locally:
```
pip install .
```

or with the extras:
```
pip install .[ruleset_factories]
```

## Extras

### Ruleset factories
Module for transforming rule-based models into decision-rules rulesets.

#### Usage
Module expose only a single function `ruleset_factory`, which is used to convert
`RuleKit`-type rulesets into `decision-rules` rulesets.

Example:
```python
from rulekit.classification import RuleClassifier

from decision_rules.ruleset_factories import ruleset_factory
from decision_rules.classification.ruleset import ClassificationRuleSet

rule_classifier = RuleClassifier()
rule_classifier.fit(X, y)

ruleset: ClassificationRuleSet = ruleset_factory(
    model=rule_classifier,
    X_train=X,
    y_train=y,
)
```

A parser for `MLRules`-type rulesets has also been implemented and can be found under
`decision_rules.ruleset_factories._factories.classification.MLRulesRuleSetFactory`.
Its `.make` method takes a list of lines read from MLRules algorithm output file,
the dataset and (optionally) a voting metric to calculate rule weights.

Example:
```python
with open("example_MLRules_output.txt") as file:
    ml_rules_lines = file.readlines()
ruleset: ClassificationRuleSet = MLRulesRuleSetFactory().make(
    ml_rules_lines, X_df, y_df, "Precision"
)
```

## Running tests
To run tests of the base package, please run:
```
python -m unittest discover ./tests/base_tests
```

To run tests of the extras modules, please run:
```
python -m unittest discover ./tests/extras
```

Similarly, you can run tests of the whole package by running:
```
python -m unittest discover ./tests
```

## Documentation
Full documentation along with some usage examples can be found [here](https://ruleminer.github.io/decision-rules/).

## License
The software is licensed under the MIT License. See the LICENSE file for details.