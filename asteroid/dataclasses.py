from pathlib import Path
import dataclasses
from typing import Literal, get_origin, get_args


def isliteral(field, value):
    return get_origin(value) is Literal


def safe_literal(field, value):
    if value not in get_args(field.type):
        raise TypeError(f"{field.name!r} must be any of {get_args(field.type)}, got {value} instead")
    return value


DATACLASS_SERIALIZE_TYPE_MAP = [
    (isliteral, lambda field, value: value, safe_literal),
    (Path, lambda field, value: str(value), lambda field, value: Path(value)),
]

def type_check(test_func, field, value):
    if isinstance(test_func, type):
        if isinstance(field.type, type):
            return issubclass(field.type, test_func)
        else:
            return False
    else:
        return test_func(field, value)


def serialize_dataclass(obj, type_map=DATACLASS_SERIALIZE_TYPE_MAP, exclude=()):
    res = {}
    for field in dataclasses.fields(obj):
        if field.name in exclude:
            continue
        value = getattr(obj, field.name)
        if dataclasses.is_dataclass(field.type):
            value = serialize_dataclass(field.type, value, type_map, exclude)
        else:
            for test_func, serialize_func, _ in type_map:
                if type_check(test_func, field, value):
                    value = serialize_func(field, value)
                    break
        res[field.name] = value
    return res


def unserialize_dataclass(cls, dct, type_map=DATACLASS_SERIALIZE_TYPE_MAP):
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in dct:
            continue
        value = dct[field.name]
        if dataclasses.is_dataclass(field.type):
            value = unserialize_dataclass(field.type, value, type_map)
        else:
            for test_func, _, unserialize_func in type_map:
                if type_check(test_func, field, value):
                    value = unserialize_func(field, value)
                    break
        kwargs[field.name] = value
    return cls(**kwargs)
