"""
Contains classed useful for serializing and deserializing objects.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Type, Union

from pydantic import BaseModel


class SerializationModes(Enum):
    """
    Specifies possible serialization modes choice.
    """
    FULL: str = 'full'
    """In this mode all important information is serialized. After 
    deserialization of the rulesets you can use it as it is without calling `update` 
    method.
    """
    MINIMAL: str = 'minimal'
    """In this mode only minimal, necessary information is serialized. After 
    deserialization of the rulesets you'll have to call `update` method to use it.
    """

    @classmethod
    def instantiate(cls, value: Union[SerializationModes, str]) -> SerializationModes:
        """Creates serialization mode instance from given value

        Args:
            value (Union[SerializationModes, str]): either serialization mode or its name

        Raises:
            ValueError: If given value is not a valid serialization mode

        Returns:
            SerializationModes: serialization mode
        """
        if isinstance(value, SerializationModes):
            return value
        try:
            return SerializationModes(value.lower())
        except ValueError as error:
            raise ValueError(
                f'Unknown serialization mode: "{value}", '
                f'expected one of: {list(cls._value2member_map_.keys())}'
            ) from error


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
                f' ({
                    cls._serializers_dict[serializable_class], serializer_class})'
            )
        cls._serializers_dict[serializable_class] = serializer_class

    @classmethod
    def serialize(
        cls,
        value: Any,
        mode: Union[str, SerializationModes] = SerializationModes.FULL
    ) -> dict:
        """Serializes given object to json dictionary

        Args:
            value (Any): value
            mode (Union[str, SerializationModes], optional): Controls serialization mode. 
                In minimal mode only minimal information is serialized. After deserialization 
                of the rulesets you'll have to call update method to use it. In full mode all
                important information is serialized and after ruleset deserialization and
                you can use it further without calling update. Defaults to SerializationModes.FULL

        Raises:
            ValueError: if no serializer class is registered for passed object type

        Returns:
            dict: json dictionary
        """
        if value is None:
            return None
        value_class = value.__class__
        if value_class not in cls._serializers_dict:
            raise ValueError(
                f'There is no registered JSONClassSerializer for class: "{value_class}"')
        return cls._serializers_dict[value_class].serialize(value, mode)

    @classmethod
    def deserialize(cls, data: dict, target_class: type) -> Any:
        """Deserializes json dictionary to given target class

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

    _Model: Type[BaseModel]

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
    def _to_pydantic_model(
        cls: type,
        instance: Any,
        mode: SerializationModes
    ) -> BaseModel:
        """Creates pydantic model from object instance

        Args:
            instance (Any): object instance
            mode (SerializationModes): Controls serialization mode. 
                In minimal mode only minimal information is serialized. In full mode all
                important information is serialized and object should be ready to use without
                calling any additional methods.

        Returns:
            BaseModel: pydantic model
        """

    @classmethod
    def serialize(
        cls,
        instance: Any,
        mode: Union[str, SerializationModes] = SerializationModes.FULL
    ) -> dict:
        """
        Args:
            instance (Any): object instance
            mode (Union[str, SerializationModes], optional): Controls serialization mode. 
                In minimal mode only minimal information is serialized. After deserialization 
                of the rulesets you'll have to call update method to use it. In full mode all
                important information is serialized and after ruleset deserialization and
                you can use it further without calling update. Defaults to SerializationModes.FULL

        Returns:
            dict: json dictionary
        """
        return cls._to_pydantic_model(
            instance, 
            mode=SerializationModes.instantiate(mode)
        ).model_dump()

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

    Example
    -------
    .. code-block:: python
        >>> class MyCustomClass(AbstractRule):
        >>>     ...
        >>>
        >>> @register_serializer(MyCustomClass)
        >>> class MyCustomRuleClassSerializer(JSONClassSerializer):
        >>>     ...

    Args:
        type (type): type of objects to be serialized
    """
    def wrapper(serializer_class):
        JSONSerializer.register_serializer(registered_type, serializer_class)
        return serializer_class
    return wrapper
