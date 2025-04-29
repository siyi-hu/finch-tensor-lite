import operator
import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Tuple, Iterable
from itertools import accumulate
from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple
from ..algebra import *

from ..finch_logic import *
from ..symbolic import gensym

@dataclass
class LazyTensor:
    data: LogicNode
    shape: Tuple
    fill_value: Any
    element_type: Any

    @property
    def ndim(self) -> int:
        return len(self.shape)

def lazy(arr) -> LazyTensor:
    """
        - lazy(arr) -> LazyTensor:
        Converts an array into a LazyTensor. If the input is already a LazyTensor, it is returned as-is.
        Otherwise, it creates a LazyTensor representation of the input array.

        Parameters:
        - arr: The input array to be converted into a LazyTensor.

        Returns:
        - LazyTensor: A lazy representation of the input array.
    """
    if isinstance(arr, LazyTensor):
        return arr
    name = Alias(gensym("A"))
    idxs = [Field(gensym("i")) for _ in range(arr.ndim)]
    shape = tuple(arr.shape)
    tns = Subquery(name, Table(Immediate(arr), idxs))
    return LazyTensor(tns, shape, fill_value(arr), element_type(arr))

def get_default_scheduler():
    return FinchLogicInterpreter()

def compute(arg, ctx=get_default_scheduler()):
    """
    - compute(arg, ctx=get_default_scheduler()):
    Executes a fused operation represented by LazyTensors. This function evaluates the entire
    operation in an optimized manner using the provided scheduler.

    Parameters:
    - arg: A lazy tensor or a tuple of lazy tensors representing the fused operation to be computed.
    - ctx: The scheduler to use for computation. Defaults to the result of `get_default_scheduler()`.

    Returns:
    - A tensor or a list of tensors computed by the fused operation.
    """
    if isinstance(arg, tuple):
        args = arg
    else:
        args = (arg,)
    vars = tuple(Alias(gensym("A")) for _ in args)
    bodies = tuple(map(lambda arg, var: Query(var, arg.data), args, vars))
    prgm = Plan(bodies + (Produces(vars),))
    res = ctx(prgm)
    if isinstance(arg, tuple):
        return tuple(res)
    else:
        return res[0]

register_property(LazyTensor, "__self__", "fill_value", lambda x: x.fill_value)
register_property(LazyTensor, "__self__", "element_type", lambda x: x.element_type)

def permute_dims(arg: LazyTensor, /, axis: Tuple[int, ...]) -> LazyTensor:
    axis = normalize_axis_tuple(axis, arg.ndim + len(axis))
    idxs = [Field(gensym("i")) for _ in range(arg.ndim)]
    return LazyTensor(
        Reorder(Relabel(arg.data, idxs), [idxs[i] for i in axis]),
        [arg.shape[i] for i in axis],
        arg.fill_value,
        arg.element_type,
    )

def identify(data):
    lhs = Alias(gensym("A"))
    return Subquery(lhs, data)

def expand_dims(
    x: LazyTensor,
    /,
    axis: int | tuple[int, ...] = 0,
) -> LazyTensor:
    if isinstance(axis, int):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, x.ndim + len(axis))
    assert len(axis) == len(set(axis)), "axis must be unique"
    assert set(axis).issubset(range(x.ndim + len(axis))), "Invalid axis"
    offset = [0] * (x.ndim + len(axis))
    for d in axis:
        offset[d] = 1
    offset = list(accumulate(offset))
    idxs_1 = [Field(gensym("i")) for _ in range(x.ndim)]
    idxs_2 = [
        Field(gensym("i")) if n in axis else idxs_1[n - offset[n]]
        for n in range(x.ndim + len(axis))
    ]
    data_2 = Reorder(Relabel(x.data, idxs_1), idxs_2)
    shape_2 = tuple(
        1 if n in axis else x.shape[n - offset[n]]
        for n in range(x.ndim + len(axis))
    )
    return LazyTensor(data_2, shape_2, x.fill_value, x.element_type)

def squeeze(
    x: LazyTensor,
    /,
    axis: int | tuple[int, ...],
) -> LazyTensor:
    if isinstance(axis, int):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, x.ndim)
    assert len(axis) == len(set(axis)), "axis must be unique"
    assert set(axis).issubset(range(x.ndim)), "Invalid axis"
    assert all(x.shape[d] == 1 for d in axis), "axis to drop must have size 1"
    newaxis = [n for n in range(x.ndim) if n not in axis]
    idxs_1 = [Field(gensym("i")) for _ in range(x.ndim)]
    idxs_2 = [idxs_1[n] for n in newaxis]
    data_2 = Reorder(Relabel(x.data, idxs_1), idxs_2)
    shape_2 = tuple(x.shape[n] for n in newaxis)
    return LazyTensor(data_2, shape_2, x.fill_value, x.element_type)

def reduce(
    op: Callable,
    x: LazyTensor,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype = None,
    keepdims: bool = False,
    init = None):
    if init is None:
        init = init_value(op, x.element_type)
    axis = normalize_axis_tuple(axis, x.ndim)
    shape = tuple(x.shape[n] for n in range(x.ndim) if n not in axis)
    fields = [Field(gensym("i")) for _ in range(x.ndim)]
    data = Aggregate(Immediate(op), Immediate(init), Relabel(x.data, fields), [fields[i] for i in axis])
    if keepdims:
        keeps = [fields[i] if i in axis else Field(gensym("j")) for i in range(x.ndim)]
        data = Reorder(data, keeps)
        shape = [shape[i] if i in axis else 1 for i in range(x.ndim)]
    if dtype is None:
        dtype = fixpoint_type(op, init, x.element_type)
    return LazyTensor(identify(data), shape, init, dtype)

def broadcast(f: Callable, src: LazyTensor, *args) -> LazyTensor:
    largs = [src, *map(lazy, args)]
    ndim = builtins.max([arg.ndim for arg in largs])
    shape = tuple(
        builtins.max([arg.shape[i - ndim + arg.ndim] if i - ndim + arg.ndim >= 0 else 1 for arg in largs])
        for i in range(ndim)
    )
    idxs = [Field(gensym("i")) for _ in range(ndim)]
    bargs = []
    for arg in largs:
        idims = []
        odims = []
        for i in range(ndim - arg.ndim,ndim):
            if arg.shape[i - ndim + arg.ndim] == shape[i]:
                idims.append(idxs[i])
                odims.append(idxs[i])
            else:
                if arg.shape[i - ndim + arg.ndim] != 1:
                    raise ValueError("Invalid shape for broadcasting")
                idims.append(Field(gensym("j")))
        bargs.append(Reorder(Relabel(arg.data, tuple(idims)), tuple(odims)))
    data = MapJoin(Immediate(f), tuple(bargs))
    new_fill_value = f(*[x.fill_value for x in largs])
    new_element_type = return_type(f, *[x.element_type for x in largs])
    return LazyTensor(identify(data), shape, new_fill_value, new_element_type)

def prod(arr: LazyTensor, dims) -> LazyTensor:
    return reduce(operator.mul, arr, dims, arr.fill_value)

def multiply(x1: LazyTensor, x2: LazyTensor) -> LazyTensor:
    return broadcast(operator.mul, x1, x2)