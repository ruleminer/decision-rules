try:
    from ._factory import ruleset_factory
except ImportError as e:
    raise ImportError(
        "The 'ruleset_factories' extra requires some additional dependencies. "
        "Please install them with: `pip install decision_rules[ruleset_factories]`"
    ) from e
