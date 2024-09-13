"""
Contains classed useful for serializing and deserializing objects.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Union

from pydantic import BaseModel


class JSONSerializer:
    """Serializes deserializes registered object types.
    It can serialize and deserialize any object for which type there was registered
    as serializer class.
    """

    _serializers_dict: dict[str, JSONClassSerializer] = {}

    @classmethod
    def register_serializer(cls, serializable_class: type, serializer_class: type):
        """Registere new serializer for given type

        Args:
            serializable_class (type): serializer class
            serializer_class (type): type of objects to be serialized / deserialized

        Raises:
            ValueError: when try to register multiple serializers for the same type
        """
        if (serializable_class in cls._serializers_dict) and \
                (serializer_class != cls._serializers_dict[serializable_class]):
            raise ValueError(
                'Trying to register multiple serializer classes for type: ' +
                f'"{serializable_class}"' +
                f' ({cls._serializers_dict[serializable_class], serializer_class})'
            )
        cls._serializers_dict[serializable_class] = serializer_class

    @classmethod
    def serialize(cls, value: Any) -> dict:
        """
        Args:
            value (Any): value

        Raises:
            ValueError: if no serializer class is registered for passed object type

        Returns:
            dict: json dictionary
        """
        value_class = value.__class__
        if value is None:
            return None
        if value_class not in cls._serializers_dict:
            raise ValueError(
                f'There is no registered JSONClassSerializer for class: "{value_class}"')
        return cls._serializers_dict[value_class].serialize(value)

    @classmethod
    def deserialize(cls, data: dict, target_class: type) -> Any:
        """
        Args:
            data (dict): json dictionary
            target_class (type): target class

        Raises:
            ValueError: if no serializer class is registered for passed object type

        Returns:
            Any: deserialized object
        """
        if target_class not in cls._serializers_dict:
            raise ValueError(
                f'There is no registered JSONClassSerializer for class: "{target_class}"')
        return cls._serializers_dict[target_class].deserialize(data)


class JSONClassSerializer(ABC):
    """Abstract class for classes serializer. Each serializer should
    inherit this class and be registered for type by "register_serializer".
    Each class serializer should have inner class "Model" which is pydantic model
    class for JSON representation.
    """

    _Model: type

    @classmethod
    @abstractmethod
    def _from_pydantic_model(cls: type, model: BaseModel) -> Any:
        """Creates object instance from pydantic model

        Args:
            model (BaseModel): pydantic model

        Returns:
            Any: object instance
        """

    @classmethod
    @abstractmethod
    def _to_pydantic_model(cls: type, instance: Any) -> BaseModel:
        """Creates pydantic model from object instance

        Args:
            instance (Any): object instance

        Returns:
            BaseModel: pydantic model
        """

    @classmethod
    def serialize(cls, instance: Any) -> dict:
        """
        Args:
            instance (Any): object instance

        Returns:
            dict: json dictionary
        """
        return cls._to_pydantic_model(instance).model_dump()

    @classmethod
    def deserialize(cls, data: Union[dict, BaseModel]) -> Any:
        """
        Args:
            data: (Union[dict, BaseModel]): dictionary or pydantic model

        Returns:
            Any: object instance
        """
        if data is None:
            return None
        if not issubclass(data.__class__, BaseModel):
            data = getattr(cls, '_Model')(**data)
        return cls._from_pydantic_model(data)


def register_serializer(
    registered_type: type
):
    """Register decorated class to be used as serializer for given type.

    Args:
        type (type): type of objects to be serialized
    """
    def wrapper(serializer_class):
        JSONSerializer.register_serializer(registered_type, serializer_class)
        return serializer_class
    return wrapper
