from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import TypedDict
from typing import Union


class SortInfo(TypedDict):
    column_name: str
    ascending: bool


class FilterOperators(Enum):
    equal: str = '='
    not_equal: str = '!='
    greater: str = '>'
    greater_equal: str = '>='
    lower: str = '<'
    lower_equal: str = '<='
    is_in: str = 'in'
    icontains: str = 'icontains'
    istartswith: str = "istartswith"


class FilterInfo(TypedDict):
    column_name: str
    operator: FilterOperators
    value: Union[Any, list[Any]]


class FilterConnector(Enum):
    AND: str = "&"
    OR: str = "|"


@dataclass
class FilterList:
    connector: FilterConnector
    filters: list[Union[FilterInfo, 'FilterList']]
