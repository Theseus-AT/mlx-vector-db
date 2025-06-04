from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Vector(_message.Message):
    __slots__ = ("values", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VALUES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, values: _Optional[_Iterable[float]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AddVectorsRequest(_message.Message):
    __slots__ = ("user_id", "model_id", "vectors")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    model_id: str
    vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    def __init__(self, user_id: _Optional[str] = ..., model_id: _Optional[str] = ..., vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ...) -> None: ...

class AddVectorsResponse(_message.Message):
    __slots__ = ("success", "count", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    count: int
    message: str
    def __init__(self, success: bool = ..., count: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ("user_id", "model_id", "query_vector", "k")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    model_id: str
    query_vector: Vector
    k: int
    def __init__(self, user_id: _Optional[str] = ..., model_id: _Optional[str] = ..., query_vector: _Optional[_Union[Vector, _Mapping]] = ..., k: _Optional[int] = ...) -> None: ...

class QueryResultItem(_message.Message):
    __slots__ = ("id", "distance", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    distance: float
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., distance: _Optional[float] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("results", "query_id")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    QUERY_ID_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[QueryResultItem]
    query_id: str
    def __init__(self, results: _Optional[_Iterable[_Union[QueryResultItem, _Mapping]]] = ..., query_id: _Optional[str] = ...) -> None: ...

class BatchQueryRequest(_message.Message):
    __slots__ = ("user_id", "model_id", "query_vectors", "k")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTORS_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    model_id: str
    query_vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    k: int
    def __init__(self, user_id: _Optional[str] = ..., model_id: _Optional[str] = ..., query_vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., k: _Optional[int] = ...) -> None: ...
