"""Ownership of array payloads in immutable semantic records.

Value records own their data. The explicit BufferResource memory-view path has
separate borrowing/mutability semantics and is deliberately not handled here.
"""

from dataclasses import fields
from typing import Any

import numpy as np
import numpy.typing as npt


def immutable_array(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Detach caller memory and return a contiguous, permanently readonly value.

    A readonly view of a writable allocation is insufficient: the owner can
    still mutate it. Immutable bytes also prevent re-enabling NumPy writes.
    Already owned immutable values can be shared by dataclass replacement.
    """
    owner: object = array
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    if isinstance(owner, bytes) and not array.flags.writeable and array.flags.c_contiguous:
        return array
    if array.dtype.hasobject:
        raise TypeError("semantic array payloads cannot contain Python objects")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def freeze_array_fields(record: Any) -> None:
    """Own direct array fields after record validation, preserving record identity."""
    for field in fields(record):
        value = getattr(record, field.name)
        if isinstance(value, np.ndarray):
            object.__setattr__(record, field.name, immutable_array(value))
        elif isinstance(value, list):
            object.__setattr__(record, field.name, tuple(value))
