import pandas as pd

from decision_rules.parsers.models import FilterConnector
from decision_rules.parsers.models import FilterList
from decision_rules.parsers.models import FilterOperators


class FilterToMaskProcessor:
    def process(self, df: pd.DataFrame, filter_list: FilterList) -> pd.Series:
        mask: pd.Series = None
        for filter_ in filter_list.filters:
            if isinstance(filter_, FilterList):
                submask = self.process(df, filter_)
            else:
                column = df[filter_['column_name']]
                submask_mapping = {
                    FilterOperators.lower: lambda: (column < filter_['value']),
                    FilterOperators.greater: lambda: (column > filter_['value']),
                    FilterOperators.lower_equal: lambda: (column <= filter_['value']),
                    FilterOperators.greater_equal: lambda: (column >= filter_['value']),
                    FilterOperators.equal: lambda: (column == filter_['value']),
                    FilterOperators.not_equal: lambda: (column != filter_['value']),
                    FilterOperators.is_in: lambda: (
                        column.isin(filter_['value'])
                    ),
                    FilterOperators.icontains: lambda: (
                        # it will fail badly on non-string columns!
                        column.astype(str).str.contains(
                            filter_['value'], case=False)
                    ),
                    FilterOperators.istartswith: lambda: (
                        # it will fail badly on non-string columns!
                        column.astype(str).str.startswith(
                            filter_['value'], case=False)
                    )
                }
                submask = submask_mapping[filter_['operator']]()
            if filter_list.connector == FilterConnector.AND:
                mask = submask if mask is None else mask & submask
            else:
                mask = submask if mask is None else mask | submask
        return mask
