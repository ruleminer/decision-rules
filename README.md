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

## Running tests
```
python -m unittest discover .\tests
```