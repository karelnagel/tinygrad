# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools, functools, struct, sys, inspect, pathlib, string, dataclasses, hashlib
from contextlib import ContextDecorator
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, cast, get_args, Literal, TYPE_CHECKING, SupportsIndex
from tinygrad.dtype import DType, DTypeLike, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype, to_dtype, truncate
from tinygrad.helpers import argfix, make_tuple, flatten, prod, all_int, round_up, merge_dicts, argsort, getenv, all_same, fully_flatten, dedup
from tinygrad.helpers import IMAGE, DEBUG, WINO, _METADATA, Metadata, TRACEMETA, ceildiv, fetch, polyN
from tinygrad.ops import smax, smin, resolve, UOp, Ops, sint, Variable, SimpleMathTrait, identity_element
from tinygrad.device import Device, Buffer, BufferSpec
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.schedule import ScheduleContext, ScheduleItem, create_schedule_with_vars, to_uop

# **** start with two base classes, Tensor and Function ****

class Function:
  def __init__(self, device:Union[str, Tuple[str, ...]], *tensors:Tensor, metadata:Optional[Metadata]=None):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors
    self.metadata = metadata

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x, metadata=_METADATA.get())
    ret = Tensor.__new__(Tensor)
    ret.lazydata, ret.requires_grad, ret.grad = ctx.forward(*[t.lazydata for t in x], **kwargs), ctx.requires_grad, None
    ret._ctx = ctx if ctx.requires_grad and not Tensor.no_grad else None  # used by autograd engine
    return ret

import tinygrad.function as F

def _metaop(op, shape:Tuple[sint,...], dtype:DType, device:Union[str, Tuple[str, ...]], arg=None, src:Tuple[LazyBuffer, ...]=()):
  if isinstance(device, str): return LazyBuffer.metaop(op, shape, dtype, device, arg, src)
  raise NotImplementedError("MultiLazyBuffer")

def get_shape(x) -> Tuple[int, ...]:
  # NOTE: str is special because __getitem__ on a str is still a str
  if not hasattr(x, "__len__") or not hasattr(x, "__getitem__") or isinstance(x, str) or (hasattr(x, "shape") and x.shape == ()): return ()
  if not all_same(subs:=[get_shape(xi) for xi in x]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(subs),) + (subs[0] if subs else ())

def _frompy(x:Union[List, Tuple, bytes], dtype:DType) -> LazyBuffer:
  if isinstance(x, bytes): ret, data = LazyBuffer.metaop(Ops.EMPTY, (len(x)//dtype.itemsize,), dtype, "PYTHON"), x
  else:
    ret = LazyBuffer.metaop(Ops.EMPTY, get_shape(x), dtype, "PYTHON")
    assert dtype.fmt is not None, f"{dtype=} has None fmt"
    truncate_function = truncate[dtype]
    data = struct.pack(f"@{ret.size}{dtype.fmt}", *[truncate_function(xi) for xi in fully_flatten(x)])
  # fake realize
  ret.buffer.allocate(memoryview(data if Device.DEFAULT != "PYTHON" else bytearray(data)))
  del ret.srcs
  return ret

def _align_left(*shapes:Tuple[sint, ...]) -> Tuple[Tuple[sint, ...], ...]:
  # unsqueeze left to make every shape same length
  max_dim = max(len(shape) for shape in shapes)
  return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)
def _broadcast_shape(*shapes:Tuple[sint, ...]) -> Tuple[sint, ...]:
  return tuple(0 if 0 in nth_dim_sizes else smax(nth_dim_sizes) for nth_dim_sizes in zip(*_align_left(*shapes)))

ReductionStr = Literal["mean", "sum", "none"]

class Tensor(SimpleMathTrait):
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
  __deletable__ = ('_ctx',)
  training: ClassVar[bool] = False
  no_grad: ClassVar[bool] = False

  def __init__(self, data:Union[None, ConstType, UOp, bytes, List, Tuple, LazyBuffer, 'np.ndarray', pathlib.Path],  # type: ignore [name-defined] # noqa: F821
               device:Optional[Union[str, tuple, list]]=None, dtype:Optional[DTypeLike]=None, requires_grad:Optional[bool]=None):
    if dtype is not None: dtype = to_dtype(dtype)
    assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
    if device is None and isinstance(data, pathlib.Path): device = f"DISK:{data.resolve()}"  # keep it on the disk if device is None
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)

    # tensors can have gradients if you have called .backward
    self.grad: Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad: Optional[bool] = requires_grad

    # internal variable used for autograd graph construction
    self._ctx: Optional[Function] = None

    # create a LazyBuffer from the different types of inputs
    if isinstance(data, (LazyBuffer)): assert dtype is None or dtype==data.dtype, "dtype doesn't match, and casting isn't supported"
    elif data is None: data = _metaop(Ops.EMPTY, (0,), dtype or dtypes.default_float, device)
    elif isinstance(data, get_args(ConstType)): data = _metaop(Ops.CONST, tuple(), dtype or dtypes.from_py(data), device, data)
    elif isinstance(data, UOp):
      assert data.op is Ops.BIND and data.src[0].op is Ops.DEFINE_VAR and data.src[1].op is Ops.CONST, f"can't create tensor from UOp {data}"
      data = _metaop(Ops.CONST, tuple(), dtype or data.dtype, device, data)
    elif isinstance(data, bytes): data = _frompy(data, dtypes.uint8 if dtype is None else dtype)
    elif isinstance(data, (list, tuple)):
      if dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool
        else: dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float
      if dtype == dtypes.bfloat16: data = Tensor(_frompy(data, dtypes.float32), device=device).cast(dtypes.bfloat16).lazydata
      else: data = _frompy(data, dtype)
    # elif str(type(data)) == "<class 'numpy.ndarray'>":
    #   import numpy as np
    #   assert isinstance(data, np.ndarray), f"expected np.ndarray, got {data}"
    #   if data.shape == (): data = _metaop(Ops.CONST, tuple(), dtype or _from_np_dtype(data.dtype), device, data.item())
    #   else: data = _fromnp(data.astype(npdtype) if dtype is not None and (npdtype:=_to_np_dtype(dtype)) is not None else data)  # type: ignore [name-defined]
    elif isinstance(data, pathlib.Path):
      dtype = dtype or dtypes.uint8
      data = _metaop(Ops.EMPTY, (data.stat().st_size // dtype.itemsize,), dtype, f"DISK:{data.resolve()}")

    # by this point, it has to be a LazyBuffer
    if not isinstance(data, (LazyBuffer)): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")

    # data might be on a different device
    if isinstance(device, str): self.lazydata:Union[LazyBuffer] = data if data.device == device else data.copy_to_device(device)
    # if device is a tuple, we should have/construct a MultiLazyBuffer
    elif isinstance(data, LazyBuffer): raise NotImplementedError("MultiLazyBuffer")
    else:
      assert data.device == device, f"MultiLazyBuffer device mismatch, {data.device} != {device}"
      self.lazydata = data

  class train(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.training = self.prev

  class test(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.no_grad = Tensor.no_grad, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.no_grad = self.prev

  def __repr__(self):
    return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad is not None else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  def __bool__(self): raise TypeError("__bool__ on Tensor is not defined")

  def __len__(self):
    if not self.shape: raise TypeError("len() of a 0-d tensor")
    return self.shape[0]

  @property
  def device(self) -> Union[str, Tuple[str, ...]]: return self.lazydata.device

  @property
  def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

  @property
  def dtype(self) -> DType: return self.lazydata.dtype

  # ***** data handlers ****

  def schedule_with_vars(self, *lst:Tensor) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
    """
    Creates the schedule needed to realize these Tensor(s), with Variables.

    NOTE: A Tensor can only be scheduled once.
    """
    schedule, var_vals = create_schedule_with_vars(flatten([x.lazydata.lbs for x in (self,)+lst]))
    return memory_planner(schedule), var_vals

  def _debug_ast(self):
    schedule,vars = create_schedule_with_vars(self.cast(self.dtype.base).contiguous().to('CLANG').lazydata.lbs)
    return [s.ast for s in schedule]
  def _debug(self):
    ctx = ScheduleContext()
    cache: Dict[LazyBuffer, UOp] = {}
    buffers: Dict[UOp, Buffer] = {}
    uop = to_uop(self.lazydata,ctx,buffers,cache)
    return uop
  def schedule(self, *lst:Tensor) -> List[ScheduleItem]:
    """Creates the schedule needed to realize these Tensor(s)."""
    schedule, var_vals = self.schedule_with_vars(*lst)
    assert len(var_vals) == 0
    return schedule

  def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:
    """Triggers the computation needed to create these Tensor(s)."""
    run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)
    return self

  def replace(self, x:Tensor) -> Tensor:
    """
    Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
    """
    # used for replacing a Tensor with a new version of it (potentially with a different device and dtype)
    assert not x.requires_grad and getattr(self, '_ctx', None) is None
    assert self.shape == x.shape, f"replace shape mismatch {self.shape} != {x.shape}"
    self.lazydata = x.lazydata
    return self

  def assign(self, x) -> Tensor:
    # TODO: this is a hack for writing to DISK. remove with working assign
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      if x.__class__ is not Tensor: x = Tensor(x, device="CLANG", dtype=self.dtype)
      self.contiguous().realize().lazydata.base.realized.copyin(x._data())
      return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.lazydata is x.lazydata: return self  # a self assign is a NOOP
    # NOTE: we allow cross device assign
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    assert self.device == x.device, f"assign device mismatch {self.device} != {x.device}"
    assert self.dtype == x.dtype, f"assign dtype mismatch {self.dtype} != {x.dtype}"
    # assert not isinstance(self.lazydata, MultiLazyBuffer) or self.lazydata.axis == x.lazydata.axis, "axis must match on MultiLazyBuffer"
    assert not x.requires_grad  # self requires_grad is okay?
    if not self.lazydata.is_realized: return self.replace(x)
    self.lazydata = self.lazydata.assign(x.lazydata)
    return self

  def detach(self) -> Tensor:
    """
    Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
    """
    return Tensor(self.lazydata, device=self.device, requires_grad=False)

  def _data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0))
    # NOTE: this realizes on the object from as_buffer being a Python object
    cpu = self.cast(self.dtype.base).contiguous().to("CLANG").realize()
    buf = cast(Buffer, cast(LazyBuffer, cpu.lazydata).base.realized)
    if self.device != "CLANG": buf.options = BufferSpec(nolru=True)
    return buf.as_buffer(allow_zero_copy=True if self.device != "CLANG" else False)

  def data(self) -> memoryview:
    """
    Returns the data of this tensor as a memoryview.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(np.frombuffer(t.data(), dtype=np.int32))
    ```
    """
    assert self.dtype.base.fmt is not None, f"no fmt dtype for {self.dtype.base}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    if TYPE_CHECKING or sys.version_info < (3, 12): assert self.dtype.base.fmt != "e"
    return cast(memoryview, self._data().cast(self.dtype.base.fmt) if 0 in self.shape else self._data().cast(self.dtype.base.fmt, self.shape))

  def item(self) -> ConstType:
    """
    Returns the value of this tensor as a standard Python number.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(42)
    print(t.item())
    ```
    """
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]

  # TODO: should be Tensor.tolist() -> Union[List[ConstType], ConstType]. The List is Sequence because mypy expects memoryview.tolist() -> list[int]
  # src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi#L803
  def tolist(self) -> Union[Sequence[ConstType], ConstType]:
    """
    Returns the value of this tensor as a nested list.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.tolist())
    ```
    """
    return self.data().tolist()

  def clone(self) -> Tensor:
    """
    Creates a clone of this tensor allocating a separate buffer for the data.
    """
    ret = Tensor(self.lazydata.clone(), self.device, requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.clone()
    if hasattr(self, '_ctx'): ret._ctx = self._ctx
    return ret
  
  def to(self, device:Optional[Union[str, Tuple[str, ...]]]) -> Tensor:
    """
    Moves the tensor to the given device.
    """
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    if device == self.device: return self
    if not isinstance(device, str): return self.shard(device)
    ret = Tensor(self.lazydata, device, requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.to(device)
    if hasattr(self, '_ctx'): ret._ctx = self._ctx
    return ret
  
  @staticmethod
  def _metaop(op, shape, device:Optional[Union[Tuple[str, ...], str]]=None, dtype:Optional[DTypeLike]=None, arg=None, **kwargs):
    dtype = to_dtype(dtype) if dtype is not None else dtypes.default_float
    return Tensor(LazyBuffer.metaop(op, shape, dtype, Device.canonicalize(device), arg), device, dtype, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs):
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    return Tensor._metaop(Ops.EMPTY, argfix(*shape), **kwargs)

  @staticmethod
  def from_url(url:str, gunzip:bool=False, **kwargs) -> Tensor:
    """
    Create a Tensor from a URL.

    This is the preferred way to access Internet resources.
    It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
    This also will soon become lazy (when possible) and not print progress without DEBUG.

    THe `gunzip` flag will gzip extract the resource and return an extracted Tensor.
    """
    return Tensor(fetch(url, gunzip=gunzip), **kwargs)

  _seed: int = int(time.time())
  _device_seeds: Dict[str, Tensor] = {}
  _device_rng_counters: Dict[str, Tensor] = {}
  @staticmethod
  def manual_seed(seed=0):
    """
    Sets the seed for random operations.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)  # reset to the same seed
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    """
    Tensor._seed, Tensor._device_seeds, Tensor._device_rng_counters = seed, {}, {}

  @staticmethod
  def _threefry_random_bits(key:Tensor, counts0:Tensor, counts1:Tensor):
    x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
    x = F.Threefry.apply(x, (key[1]._broadcast_to(x.shape).cast(dtypes.uint64) << 32) | key[0]._broadcast_to(x.shape).cast(dtypes.uint64))
    counts0, counts1 = (x & 0xffffffff).cast(dtypes.uint32), ((x >> 32) & 0xffffffff).cast(dtypes.uint32)
    return counts0.cat(counts1)

  @staticmethod
  def rand(*shape, device:Optional[str]=None, dtype:Optional[DTypeLike]=None, contiguous:bool=True, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    if not dtypes.is_float(dtype := to_dtype(dtype or dtypes.default_float)): raise ValueError(f"rand only supports float dtypes, got {dtype}")
    if not all_int(shape:=argfix(*shape)) or not all(s >= 0 for s in shape): raise ValueError(f"invalid input {shape=}")
    if device is not None and not isinstance(device, str): raise ValueError(f"rand only supports single device, got {device=}")
    _device = device = Device.canonicalize(device)

    # when using MOCKGPU and NV generate rand on CLANG
    if getenv("MOCKGPU") and device.startswith("NV"): device = "CLANG"

    # generate per device seeds and rng counter if we haven't seen this device yet
    if device not in Tensor._device_seeds:
      Tensor._device_seeds[device] = Tensor(
        [int.from_bytes(hashlib.sha256(len(Tensor._device_seeds).to_bytes(4, "big")).digest(), "big"), Tensor._seed],
        device=device, dtype=dtypes.uint32, requires_grad=False)
      Tensor._device_rng_counters[device] = Tensor([0], device=device, dtype=dtypes.uint32, requires_grad=False)
      had_counter = False
    else: had_counter = True

    # if shape has 0, return zero tensor
    if (numel := prod(shape)) == 0: return Tensor.zeros(shape, device=_device, dtype=dtype, **kwargs)
    num = ceildiv(numel * dtype.itemsize, 4)

    # increment rng counter for devices
    if had_counter: Tensor._device_rng_counters[device].assign(Tensor._device_rng_counters[device] + num).contiguous()

    # threefry random bits
    counts0 = (Tensor.arange(ceildiv(num, 2), device=device, dtype=dtypes.uint32, requires_grad=False)+Tensor._device_rng_counters[device])
    counts1 = counts0 + ceildiv(num, 2)
    bits = Tensor._threefry_random_bits(Tensor._device_seeds[device], counts0, counts1)[:num]

    # bitcast to uint with same number of bits
    _, nmant = dtypes.finfo(dtype)
    uint_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[dtype.itemsize]
    bits = bits.bitcast(uint_dtype)
    # only randomize the mantissa bits and set the exponent to 1
    one = Tensor.ones_like(bits, device=bits.device, dtype=dtype).bitcast(uint_dtype)
    bits = bits.rshift((dtype.itemsize * 8) - nmant).bitwise_or(one)
    # bitcast back to the original dtype and reshape
    out = bits.bitcast(dtype)[:numel].sub(1).reshape(shape)

    # move back to the original device if we were using MOCKGPU
    if getenv("MOCKGPU") and _device: out = out.to(_device)

    out.requires_grad = kwargs.get("requires_grad")
    return out.contiguous() if contiguous else out

  # ***** creation helper functions *****

  @staticmethod
  def full(shape:Tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)

  @staticmethod
  def zeros(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 0.0, **kwargs)

  @staticmethod
  def ones(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 1.0, **kwargs)

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> Tensor:
    """
    Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.

    If `stop` is not specified, values are generated from `[0, start)` with the given `step`.

    If `stop` is specified, values are generated from `[start, stop)` with the given `step`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5.5, 10, 2).numpy())
    ```
    """
    if stop is None: stop, start = start, 0
    dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    # NOTE: this matches numpy, torch raises RuntimeError if stop-start and step have different signs
    if (output_len:=ceildiv(stop-start, step)) <= 0: return Tensor([], dtype=dtype, **kwargs)
    return (Tensor.full((output_len,), step, dtype=dtype, **kwargs)._cumalu(0, Ops.ADD) + (start - step)).cast(dtype)

  def full_like(self, fill_value:ConstType, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with the given value.
    If `dtype` is not specified, the dtype of `self` is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.full_like(t, 42).numpy())
    ```
    """
    return Tensor.full(self.shape, fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)


  def ones_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 3)
    print(Tensor.ones_like(t).numpy())
    ```
    """
    return self.full_like(1, **kwargs)

  # ***** rng hlops *****

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randint(2, 3, low=5, high=10).numpy())
    ```
    """
    if not isinstance(low, int) or not isinstance(high, int): raise TypeError(f"{low=} and {high=} must be integers")
    dtype = to_dtype(kwargs.pop("dtype", dtypes.int32))
    if not dtypes.is_int(dtype): raise TypeError(f"{dtype=} must be int")
    return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)


  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.uniform(2, 3, low=2, high=10).numpy())
    ```
    """
    dtype = kwargs.pop("dtype", dtypes.default_float)
    return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low


  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    """
    <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.glorot_uniform(2, 3).numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(argfix(*shape)[0]+prod(argfix(*shape)[1:])))**0.5)

  # ***** toposort and backward pass *****

  def _deepwalk(self):
    def _walk(node, visited):
      visited.add(node)
      # if tensor is not leaf, reset grad
      if (ctx := getattr(node, "_ctx", None)) is not None and len(ctx.parents) != 0: node.grad = None
      if ctx:
        for i in node._ctx.parents:
          if i not in visited: yield from _walk(i, visited)
        yield node
    return list(_walk(self, set()))

  def backward(self, gradient:Optional[Tensor]=None, retain_graph:bool=False) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    If 'retain_graph' is false, the graph used to compute the grads will be freed. Otherwise, it will be kept. Keeping it can increase memory usage.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    toposorted = self._deepwalk()
    if gradient is None:
      assert self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
      # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
      # this is "implicit gradient creation"
      gradient = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)

    assert self.shape == gradient.shape, f"grad shape must match tensor shape, {gradient.shape!r} != {self.shape!r}"
    self.grad = gradient
    for t0 in reversed(toposorted):
      if t0.grad is None: raise RuntimeError(f"tensor {t0} has no grad")
      token = _METADATA.set(dataclasses.replace(md, backward=True) if (md := t0._ctx.metadata) is not None else None)
      grads = t0._ctx.backward(t0.grad.lazydata)
      _METADATA.reset(token)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      if not retain_graph: del t0._ctx
    return self

  # ***** movement low level ops *****

  def view(self, *shape) -> Tensor:
    """`.view` is an alias for `.reshape`."""
    return self.reshape(shape)

  def reshape(self, shape, *args) -> Tensor:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    # resolve None and args
    new_shape = tuple([s if s is not None else self.shape[i] for i,s in enumerate(argfix(shape, *args))])
    # resolve -1
    if (c := new_shape.count(-1)) > 1: raise RuntimeError(f"only one dimension can be inferred using -1, getting {new_shape}")
    if c: new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape])
    return F.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self

  def expand(self, shape, *args) -> Tensor:
    """
    Returns a tensor that is expanded to the shape that is specified.
    Expand can also increase the number of dimensions that a tensor has.

    Passing a `-1` or `None` to a dimension means that its size will not be changed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.expand(4, -1).numpy())
    ```
    """
    new_shape = tuple(from_ if to == -1 or to is None else to for from_, to in zip(*(_align_left(self.shape, argfix(shape, *args)))))
    return self._broadcast_to(new_shape)

  def permute(self, order, *args) -> Tensor:
    """
    Returns a tensor that is a permutation of the original tensor.
    The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
    `order` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.permute(1, 0).numpy())
    ```
    """
    order_arg = tuple(self._resolve_dim(x) for x in argfix(order, *args))
    if sorted(order_arg) != list(range(self.ndim)): raise RuntimeError(f"order is not a valid permutation, getting {order_arg}")
    return F.Permute.apply(self, order=order_arg)

  def flip(self, axis, *args) -> Tensor:
    """
    Returns a tensor that reverses the order of the original tensor along given `axis`.
    `axis` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip((0, 1)).numpy())
    ```
    """
    axis_arg = tuple(self._resolve_dim(x) for x in argfix(axis, *args))
    if len(axis_arg) != len(dedup(axis_arg)): raise RuntimeError(f"dim can appear at most once, getting {axis_arg}")
    return F.Flip.apply(self, axis=axis_arg)

  def shrink(self, arg:Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor:
    """
    Returns a tensor that shrinks the each axis based on input arg.
    `arg` must have the same length as `self.ndim`.
    For each axis, it can be `None`, which means no shrink, or a tuple `(start, end)` that works the same as Python slice.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink(((None, (1, 3)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink((((0, 2), (0, 2)))).numpy())
    ```
    """
    if (shrink_arg:=[x if x is not None else (0,s) for x,s in zip(arg, self.shape)]) == [(0,s) for s in self.shape]: return self
    return F.Shrink.apply(self, arg=tuple(shrink_arg))

  def pad(self, padding:Union[Sequence[sint], Sequence[Optional[Tuple[sint, sint]]]], mode:str="constant", value:float=0.0) -> Tensor:
    """
    Returns a tensor with padding applied based on the input `padding`.
    `padding` supports two padding structures:

    1. Flat padding: (padding_left, padding_right, padding_top, padding_bottom, ...)
       - This structure matches PyTorch's pad.
       - `padding` length must be even.

    2. Group padding: (..., (padding_top, padding_bottom), (padding_left, padding_right))
       - This structure matches pad for jax, numpy, tensorflow and others.
       - For each axis, padding can be `None`, meaning no padding, or a tuple `(start, end)`.
       - `padding` must have the same length as `self.ndim`.

    Padding values can be negative, resulting in dimension shrinks that work similarly to Python negative slices.
    Padding modes is selected with `mode` which supports `constant`, `reflect` and `replicate`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1)).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad(((None, None, (0, -1), (1, 2)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad((1, 2, 0, -1), value=-float('inf')).numpy())
    ```
    """
    if mode not in {"constant", "reflect", "replicate", "circular"}: raise NotImplementedError(f"{mode=} is not supported")
    if (flat:=all(isinstance(p, (int,UOp)) for p in padding)) and len(padding)%2 != 0: raise ValueError("Flat padding must have even number of pads")
    # turn flat padding into group padding
    pX = ((0,0),)*(self.ndim - len(padding)//2) + tuple(zip(padding[-2::-2], padding[::-2])) if flat else padding
    if len(pX) != self.ndim: raise ValueError(f"padding length is improper, {padding=} {self.ndim=}")
    X, pX = self, cast(Tuple[Tuple[sint, sint]], tuple((0,0) if p is None else p for p in pX))
    pads = tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX)
    if mode == "constant":
      def _constant(x,px,v): return F.Pad.apply(x, arg=px) if v == 0 else F.Pad.apply(x, arg=px) + F.Pad.apply(Tensor.ones_like(x), arg=px).where(0,v)
      return _constant(X, pX, value) if all(resolve(p >= 0) for p in flatten(pX)) else \
             _constant(X.shrink(tuple((-smin(pB,0),smin(pA+s,s)) for (pB,pA),s in zip(pX, X.shape))), pads, value)
    # TODO:not needed for mnist
    # assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    # if mode == "circular":
    #   if any(pB>sh or pA>sh for (pB,pA),sh in zip(pX, X.shape)): raise ValueError('Padding value causes wrapping around more than once.')
    #   if any(pB<0 or pA<0 for pB,pA in pX): raise NotImplementedError("Negative pads with circular pads is not supported")
    #   orig_shape, X = X.shape, X.repeat(tuple(1 + bool(pB) + bool(pA) for pB,pA in pads))
    #   return X.shrink(tuple((0 if pB == 0 else osh-pB, xsh if pA == 0 else xsh-osh+pA) for (pB,pA),osh,xsh in zip(pads, orig_shape, X.shape)))
    # for d,(pB,pA) in enumerate(pads):
    #   if mode == "reflect":
    #     if pB >= (s:=X.shape[d]) or pA>=s: raise ValueError(f"Padding ({pB}, {pA}) should be less than the input size={s} for dim={d}.")
    #     slcB, slcA, = slice(pB,0,-1), slice(s-2 if s-2>=0 else None, s-2-pA if s-2-pA>=0 else None, -1)
    #     xB, xA = (X[[slc if i == d else slice(None) for i in range(X.ndim)]] if p > 0 else None for slc, p in ((slcB, pB), (slcA, pA)))
    #   if mode == "replicate":
    #     shrB, shrA, = tuple((0,1) if i==d else None for i in range(X.ndim)), tuple((X.shape[i]-1,X.shape[i]) if i==d else None for i in range(X.ndim))
    #     xB, xA = (X.shrink(shr).expand(tuple(p if i==d else None for i in range(X.ndim))) if p > 0 else None for shr, p in ((shrB, pB), (shrA, pA)))
    #   X = Tensor.cat(*(X_ for X_ in (xB, X, xA) if X_ is not None), dim=d)
    # return X.shrink(tuple((-min(pB,0), min(pA+s,s)) for (pB,pA),s in zip(pX, X.shape)))

  # ***** movement high level ops *****

  # Supported Indexing Implementations:
  #   1. Int indexing (no copy)
  #     - for all dims where there's int, shrink -> reshape
  #     - negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  #     - X = Tensor.rand(4,5,9); X[2,-2] shrinks the Tensor to X.shrink(((2, 3), (3, 4), (0, 9))) -> X.shape=(1,1,9)
  #     - Then we reshape (collapse) the int dim away such that for X: (1,1,9) -> (9,)
  #   2. Slice indexing (no copy)
  #     - for all dims where slice is start:end:stride, shrink -> Optional[flip] -> pad -> reshape -> shrink
  #     - first shrink the Tensor to X.shrink(((start, end),))
  #     - then we apply stride through Optional[flip] -> pad -> reshape -> shrink
  #       - flip where dim value is negative
  #       - pad on dims to be multiple of strides, such that reshaping [dim_size_padded] -> [dim_size_padded // stride, stride] is possible
  #       - shrink [dim_size_padded // stride, stride] -> [dim_size_padded // stride, 1]
  #       - reshape [dim_size_padded // stride, 1] -> [dim_size_padded // stride] and now you have your stride
  #   3. None indexing (no copy)
  #     - reshape (inject) a dim at the dim where there's None
  #   4. Tensor indexing (copy)
  #     - use Tensor.arange == tensor_index to create masks for dims with Tensors (adds a dim for each mask)
  #     - combine masks together with mul
  #     - apply mask to self by mask * self
  #     - sum reduce away the extra dims added from creating masks
  # Tiny Things:
  #   1. Supported indices: Union[int, slice, Tensor, None, List, Tuple, Ellipsis]
  #     - for any list, List[Union[List, Tuple, int]], must have homogeneous shape
  #     - for any tuple, Tuple[Union[List, Tuple, int]], must have homogeneous shape
  #   2. Bool indexing is not supported
  #   3. Out of bounds Tensor indexing results in 0
  #     - e.g: Tensor([1, 2, 3])[Tensor([4, 3, 2])] -> [0, 0, 3] index 4 and 3 are out of bounds
  def _getitem(self, indices, v: Optional[Tensor] = None) -> Tensor:
    # wrap single index into a list
    if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)): indices = [indices]
    # turn scalar Tensors into const val for int indexing if possible
    x, indices = self, [self._to_const_val(i) if isinstance(i, Tensor) and i.shape == () else i for i in indices]

    # filter ellipsis and fill with slice(None) or fill rest of indices with slice(None)
    if len(ellipsis_idx := [dim for dim, i in enumerate(indices) if i is Ellipsis]) > 1: raise IndexError("indices can only have a single ellipsis")
    fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
    num_indices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i is None)
    if num_indices > self.ndim: raise IndexError(f"too many {num_indices=} for {self.ndim=}")
    indices[fill_idx:fill_idx+1] = [slice(None)] * (self.ndim - num_indices)

    indices_parsed, dim = [], 0
    for index in indices:
      size = 1 if index is None else self.shape[dim]
      boundary, stride = [0, size], 1  # defaults
      match index:
        case list() | tuple() | Tensor():
          if not isinstance(index, Tensor): index = Tensor(index, self.device, requires_grad=False)
          if not dtypes.is_int(index.dtype): raise IndexError(f"index dtype {index.dtype} is not supported")
          index = (index.to(self.device) < 0).where(size, 0) + index # treat negative index values
        case int() | UOp(): # sint
          if index >= size or index < -size: raise IndexError(f"{index=} is out of bounds with {size=}")
          boundary = [index, index+1] if index >= 0 else [index+size, index+size+1]
        case slice():
          if index.step == 0: raise ValueError(f"{index=} cannot have 0 as step")
          if not all(isinstance(s,int) or s is None for s in (index.start,index.stop,index.step)): raise TypeError("only int slicing is supported")
          # handle int slicing
          *boundary, stride = index.indices(cast(SupportsIndex, size))
          if stride * (boundary[1] - boundary[0]) < 0: boundary = [0, 0]
          elif stride < 0: boundary = [boundary[1] + 1, boundary[0] + 1]
          # update size for slice
          size = ceildiv((boundary[1] - boundary[0]), abs(stride))
        case None: pass # do nothing
        case _: raise IndexError(f"{type(index).__name__} indexing is not supported")
      indices_parsed.append({"index":index, "size":size, "boundary":tuple(boundary), "stride":stride})
      if index is not None: dim += 1

    # movement op indexing
    if mops := [i for i in indices_parsed if i['index'] is not None]:
      # flip negative strides
      shrinks, strides = zip(*((i['boundary'], i['stride']) for i in mops))
      x = x.shrink(shrinks).flip(tuple(i for i,st in enumerate(strides) if st < 0))
      # handle stride != 1 or -1
      if any(abs(st) != 1 for st in strides):
        strides = tuple(abs(s) for s in strides)
        # pad shape to multiple of stride
        if not all_int(x.shape): raise RuntimeError("symbolic shape not supprted")
        x = x.pad(tuple((0, round_up(s, st) - s) for s, st in zip(x.shape, strides)))
        x = x.reshape(tuple(flatten((s // st, st) for s, st in zip(x.shape, strides))))
        x = x.shrink(tuple(flatten(((0, s), (0, 1)) for s in x.shape[::2]))).reshape(x.shape[::2])

    # dim injection from None by including None dim size (which is 1) and dim collapse by skipping int dim size
    x = x.reshape(tuple(index['size'] for index in indices_parsed if not isinstance(index['index'], int)))

    # tensor indexing
    if tops := [(d,i) for d,i in enumerate(i_ for i_ in indices_parsed if not isinstance(i_['index'], int)) if isinstance(i['index'], Tensor)]:
      # unload the tensor object into actual tensors
      dims, tensors, masks = [d for d,_ in tops], cast(list[Tensor], [i['index'] for _,i in tops]), []
      pre_reduce_shape = x.shape[:dims[0]] + (big_shape := _broadcast_shape(*(t.shape for t in tensors))) + x.shape[dims[0]:]

      # create index masks
      for dim, tensor in zip(dims, tensors):
        try: i = tensor.reshape(tensor.shape + (1,)*(x.ndim - dims[0])).expand(pre_reduce_shape)
        except ValueError as e: raise IndexError(f"cannot broadcast indices: {e}") from e
        masks.append(i._one_hot_along_dim(num_classes=x.shape[dim], dim=(dim - x.ndim)))

      # reduce masks to 1 mask
      mask: Tensor = functools.reduce(lambda x,y: x.mul(y), masks)

      # inject 1's for the extra dims added in create masks
      reshape_arg = x.shape[:dims[0]] + (1,) * len(big_shape) + x.shape[dims[0]:]
      # sum reduce the extra dims introduced in create masks
      x = (x.reshape(reshape_arg) * mask).sum(sum_axis:=tuple(d + len(big_shape) for d in dims), acc_dtype=x.dtype)

      # special permute case
      if dims[0] != 0 and len(dims) != 1 and tuple(dims) != tuple(range(dims[0], dims[-1]+1)):
        x = x.permute(*range(dims[0], dims[0]+len(big_shape)), *range(0, dims[0]), *range(dims[0]+len(big_shape), x.ndim))

      # for advanced setitem, returns whole tensor with indices replaced
      # TODO:not needed for mnist
      # if v is not None:
      #   vb = v.cast(self.dtype)._broadcast_to(_broadcast_shape(x.shape, v.shape))
      #   # add back reduced dims from sum
      #   for dim in sum_axis: vb = vb.unsqueeze(dim)
      #   # run _masked_setitem on tuple of axis that is to be reduced to match self.shape
      #   x = _masked_setitem(self, vb, mask, tuple(range(dims[0], dims[0] + len(big_shape))))

    return x

  def __getitem__(self, indices) -> Tensor:
    return self._getitem(indices)


  def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    """
    Concatenates self with other `Tensor` in `args` along an axis specified by `dim`.
    All tensors must have the same shape except in the concatenating dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
    print(t0.cat(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.cat(t1, t2, dim=1).numpy())
    ```
    """
    dim = self._resolve_dim(dim)
    for arg in args: assert arg.ndim==self.ndim and all(ti==ai for i,(ti,ai) in enumerate(zip(self.shape, arg.shape)) if i!=dim)
    tensors = [self, *args]
    dim_cumsum = list(itertools.accumulate([t.shape[dim] for t in tensors], initial=0))
    for i,t in enumerate(tensors): tensors[i] = t.pad([(dim_cumsum[i], dim_cumsum[-1]-dim_cumsum[i+1]) if j==dim else None for j in range(t.ndim)])
    return functools.reduce(Tensor.add, tensors)


  def repeat(self, repeats, *args) -> Tensor:
    """
    Repeats tensor number of times along each dimension specified by `repeats`.
    `repeats` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.repeat(4, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.repeat(4, 2, 1).shape)
    ```
    """
    repeats = argfix(repeats, *args)
    base_shape = _align_left(self.shape, repeats)[0]
    unsqueezed_shape = flatten([[1, s] for s in base_shape])
    expanded_shape = flatten([[r, s] for r,s in zip(repeats, base_shape)])
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return self.reshape(unsqueezed_shape).expand(expanded_shape).reshape(final_shape)

  def _resolve_dim(self, dim:int, *, extra:bool=False) -> int:
    total = self.ndim + int(extra)
    if not -max(1, total) <= dim <= max(1, total)-1: raise IndexError(f"{dim=} out of range {[-max(1, total), max(1, total)-1]}")
    return dim + total if dim < 0 else dim

  @property
  def T(self) -> Tensor:
    """`.T` is an alias for `.transpose()`."""
    return self.transpose()

  def transpose(self, dim0=1, dim1=0) -> Tensor:
    """
    Returns a tensor that is a transposed version of the original tensor.
    The given dimensions `dim0` and `dim1` are swapped.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.transpose(0, 1).numpy())
    ```
    """
    order = list(range(self.ndim))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return self.permute(order)

  def flatten(self, start_dim=0, end_dim=-1):
    """
    Flattens the tensor by reshaping it into a one-dimensional tensor.
    If `start_dim` or `end_dim` are passed, only dimensions starting with `start_dim` and ending with `end_dim` are flattened.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(8).reshape(2, 2, 2)
    print(t.flatten().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flatten(start_dim=1).numpy())
    ```
    """
    start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])

  # ***** reduce ops *****

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False) -> Tensor:
    axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
    if self.ndim == 0: axis = ()
    ret = fxn.apply(self, axis=axis)
    return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis))

  def sum(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, acc_dtype:Optional[DTypeLike]=None):
    """
    Returns the sum of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
    If not specified, the accumulation data type is chosen based on the input tensor's data type.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=1).numpy())
    ```
    """
    ret = self.cast(sum_acc_dtype(self.dtype) if acc_dtype is None else acc_dtype)._reduce(F.Sum, axis, keepdim)
    return ret.cast(self.dtype) if acc_dtype is None and self.dtype in (dtypes.float16, dtypes.bfloat16) else ret

  def prod(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, acc_dtype:Optional[DTypeLike]=None):
    """
    Returns the product of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
    If not specified, the accumulation data type is chosen based on the input tensor's data type.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, -2, -3, 1, 2, 3]).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod(axis=1).numpy())
    ```
    """
    return self.cast(acc_dtype if acc_dtype is not None else self.dtype)._reduce(F.Prod, axis, keepdim)

  def max(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Returns the maximum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=1, keepdim=True).numpy())
    ```
    """
    return self._reduce(F.Max, axis, keepdim)

  def min(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Returns the minimum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=1, keepdim=True).numpy())
    ```
    """
    if dtypes.is_int(self.dtype) or self.dtype == dtypes.bool: return ~((~self).max(axis=axis, keepdim=keepdim))
    return -((-self).max(axis=axis, keepdim=keepdim))

  def any(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Tests if any element evaluates to `True` along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[True, True], [True, False], [False, False]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any(axis=1, keepdim=True).numpy())
    ```
    """
    return self.bool().max(axis, keepdim)

  def all(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Tests if all element evaluates to `True` along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[True, True], [True, False], [False, False]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all(axis=1, keepdim=True).numpy())
    ```
    """
    return self.logical_not().any(axis, keepdim).logical_not()

  def mean(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False):
    """
    Returns the mean value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the mean is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=1).numpy())
    ```
    """
    output_dtype = self.dtype if dtypes.is_float(self.dtype) else dtypes.float32
    numerator = self.cast(sum_acc_dtype(self.dtype)).sum(axis=axis, keepdim=keepdim)
    return numerator.div(prod([si for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])).cast(output_dtype)

  def std(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, correction=1):
    """
    Returns the standard deviation of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the standard deviation is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=1).numpy())
    ```
    """
    return self.var(axis, keepdim, correction).sqrt()

  def std_mean(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, correction=1):
    """
    Calculates the standard deviation and mean over the dimensions specified by dim.
    Syntactic sugar around `Tensor.std` and `Tensor.mean` to match `torch.std_mean`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    std, mean = t.std_mean()
    print(std.numpy(), mean.numpy())
    ```
    """
    return self.std(axis, keepdim, correction), self.mean(axis, keepdim)

  def _softmax(self, axis, dtype:Optional[DTypeLike]=None):
    x = self.cast(dtype) if dtype is not None else self
    m = x - x.max(axis=axis, keepdim=True).detach()
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1, dtype:Optional[DTypeLike]=None):
    """
    Applies the softmax function to the tensor along the specified axis.

    Rescales the elements of the tensor such that they lie in the range [0, 1] and sum to 1.

    You can pass in the `axis` keyword argument to control the axis along which the softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax(axis=0).numpy())
    ```
    """
    _, e, ss = self._softmax(axis, dtype)
    return e.div(ss)

  def log_softmax(self, axis=-1, dtype:Optional[DTypeLike]=None):
    """
    Applies the log-softmax function to the tensor along the specified axis.

    The log-softmax function is a numerically stable alternative to the softmax function in log space.

    You can pass in the `axis` keyword argument to control the axis along which the log-softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax(axis=0).numpy())
    ```
    """
    m, _, ss = self._softmax(axis, dtype)
    return m - ss.log()

  def logsumexp(self, axis=None, keepdim=False):
    """
    Computes the log-sum-exp of the tensor along the specified axis or axes.

    The log-sum-exp function is a numerically stable way to compute the logarithm of the sum of exponentials.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the log-sum-exp is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=1).numpy())
    ```
    """
    m = self.max(axis=axis, keepdim=True)
    return (self - m).exp().sum(axis=axis, keepdim=keepdim).log() + m.squeeze(axis)

  def logcumsumexp(self, axis=0):
    """
    Computes the log-cumsum-exp of the tensor along the specified axis or axes.

    The log-cumsum-exp function is a numerically stable way to compute the logarithm of the cumulative sum of exponentials.

    You can pass in the `axis` keyword argument to control the axis along which
    the log-cum-sum-exp is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logcumsumexp(axis=1).numpy())
    ```
    """
    m = self.max(axis=axis, keepdim=True)
    return (self - m).exp().cumsum(axis=axis).log() + m

  def argmax(self, axis=None, keepdim=False):
    """
    Returns the indices of the maximum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax().numpy()) # Returns the index of the maximum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=0).numpy()) # Returns the indices of the maximum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=1).numpy()) # Returns the indices of the maximum values along axis 1.
    ```
    """
    if axis is None: return self.flatten().argmax(0)
    axis = self._resolve_dim(axis)
    m = self == self.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(self.shape[axis],0,-1, requires_grad=False, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return (self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)).cast(dtypes.int32)

  def argmin(self, axis=None, keepdim=False):
    """
    Returns the indices of the minimum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin().numpy()) # Returns the index of the minimum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=0).numpy()) # Returns the indices of the minimum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=1).numpy()) # Returns the indices of the minimum values along axis 1.
    ```
    """
    return (-self).argmax(axis=axis, keepdim=keepdim)

  # ***** processing ops *****

  def _pool(self, k_:Tuple[sint, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    s_, d_ = make_tuple(stride, len(k_)), make_tuple(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop, i_ = [None] * (self.ndim-len(k_)), self.shape[-len(k_):]
    assert all(resolve(d*(k-1)+1 <= i) for k,d,i in zip(k_,d_,i_)), "kernel size cannot be greater than actual input size"
    o_ = [ceildiv(i-d*(k-1), s) for i,d,k,s in zip(i_,d_,k_,s_)]
    if any(resolve(k > s) for k,s in zip(k_,s_)) or any(d != 1 for d in d_):
      # input size scaling factor to make sure shrink for stride is possible
      f_ = [1 + int(resolve(o*s > i+d)) for o,s,i,d in zip(o_,s_,i_,d_)]
      # # repeats such that we don't need padding
      x = self.repeat([1]*len(noop) + [ceildiv(k*(i*f+d),i) for k,i,d,f in zip(k_,i_,d_,f_)])
      # handle dilation
      x = x.shrink(tuple(noop + [(0,k*(i*f+d)) for k,i,d,f in zip(k_,i_,d_,f_)])).reshape(noop + flatten((k,(i*f+d)) for k,i,d,f in zip(k_,i_,d_,f_)))
      # handle stride
      x = x.shrink(tuple(noop + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_,o_,s_)))).reshape(noop + flatten((k,o,s) for k,o,s in zip(k_,o_,s_)))
      x = x.shrink(tuple(noop + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_,o_)))).reshape(noop + flatten((k,o) for k,o in zip(k_,o_)))
      # permute to move reduce to the end
      return x.permute(*range(len(noop)), *[len(noop)+i*2+1 for i in range(len(i_))], *[len(noop)+i*2 for i in range(len(i_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation
    x = self.pad(tuple(noop + [(0, max(0,o*s-i)) for i,o,s in zip(i_,o_,s_)])).shrink(tuple(noop + [(0,o*s) for o,s in zip(o_,s_)]))
    x = x.reshape(noop + flatten(((o,s) for o,s in zip(o_,s_))))
    x = x.shrink(tuple(noop + flatten(((0,o), (0,k)) for o,k in zip(o_,k_))))
    return x.permute(*range(len(noop)), *[len(noop)+i*2 for i in range(len(i_))], *[len(noop)+i*2+1 for i in range(len(i_))])

  def _padding2d(self, padding:Union[int, Sequence[int]], dims:int) -> Sequence[int]:
    return [padding]*2*dims if isinstance(padding, int) else (padding if len(padding) == 2*dims else [p for p in padding for _ in range(2)][::-1])

  def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1, padding=0):
    """
    Applies max pooling over a tensor.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    See: https://paperswithcode.com/method/max-pooling

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(25).reshape(1, 1, 5, 5)
    print(t.max_pool2d().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max_pool2d(padding=1).numpy())
    ```
    """
    padding_ = self._padding2d(padding, len(k_ := make_tuple(kernel_size, 2)))
    return self.pad(padding_, value=dtypes.min(self.dtype))._pool(k_, stride if stride is not None else k_, dilation).max(tuple(range(-len(k_), 0)))

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding:int|Tuple[int, ...]=0,
             acc_dtype:Optional[DTypeLike]=None) -> Tensor:
    """
    Applies a convolution over a tensor with a given `weight` and optional `bias`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv2d(w).numpy())
    ```
    """
    if IMAGE: return self.image_conv2d(weight, bias, groups, stride, dilation, padding, acc_dtype)
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"  # noqa: E501
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"  # noqa: E501
    padding_ = self._padding2d(padding, len(HW))

    # conv2d is a pooling op (with padding)
    x = self.pad(padding_)._pool(HW, stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not WINO:
      # normal conv
      x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True, acc_dtype=acc_dtype).reshape(bs, cout, *oyx)  # noqa: E501
      return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

    # TODO: not needed for mnist
    # HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    # winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    # winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    # winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # # todo: stride == dilation
    # # use padding to round up to 4x4 output tiles
    # # (bs, cin_, tyx, HWI)
    # d = self.pad(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for i, dim in enumerate(self.shape[-len(HW):])], []))._pool(HWI, HWO)  # noqa: E501
    # # move HW to the front: # (HWI, bs, cin_, tyx)
    # d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
    # tyx = d.shape[-len(HWI):]  # dim of tiling

    # g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  # move HW to the front

    # # compute 6x6 winograd tiles: GgGt, BtdB
    # # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    # gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    # dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)

    # # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    # ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), acc_dtype=acc_dtype), len(HW))

    # # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    # ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    # ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    # return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()


  def dot(self, w:Tensor, acc_dtype:Optional[DTypeLike]=None) -> Tensor:

    """
    Performs dot product between two tensors.
    If `w` is 1-D, it's a sum product over the last axis of `self` and `w`.
    If `w` is N-D with N>=2, it's a sum product over the last axis of `self` and the second-to-last axis of `w`.

    You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 0])
    print(a.dot(b).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.dot(b).numpy())
    ```
    """
    if IMAGE: return self.image_dot(w, acc_dtype)
    x, dx, dw = self, self.ndim, w.ndim
    if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
    if x.shape[-1] != w.shape[axis_w:=-min(w.ndim,2)]: raise RuntimeError(f"cannot dot {x.shape} and {w.shape}")
    x = x.reshape(*x.shape[0:-1], *[1]*min(dx-1, dw-1, 1), x.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:]).transpose(-1, axis_w)
    return (x*w).sum(-1, acc_dtype=acc_dtype).cast(least_upper_dtype(x.dtype, w.dtype) if acc_dtype is None else acc_dtype)

  def matmul(self, x:Tensor, reverse=False, acc_dtype:Optional[DTypeLike]=None) -> Tensor:
    """
    Performs matrix multiplication between two tensors.

    You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
    You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.matmul(b).numpy())
    ```
    """
    return x.dot(self, acc_dtype=acc_dtype) if reverse else self.dot(x, acc_dtype=acc_dtype)

  def _cumalu(self, axis:int, op:Ops, _include_initial=False) -> Tensor:
    assert self.shape[axis] != 0 and op in (Ops.ADD, Ops.MAX)
    pl_sz = self.shape[axis] - int(not _include_initial)
    pooled = self.transpose(axis,-1).pad((pl_sz, -int(_include_initial)), value=identity_element(op, self.dtype))._pool((self.shape[axis],))
    return (pooled.sum(-1) if op is Ops.ADD else pooled.max(-1)).transpose(axis,-1)

  def cumsum(self, axis:int=0) -> Tensor:
    """
    Computes the cumulative sum of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumsum(1).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.ADD)

  def cummax(self, axis:int=0) -> Tensor:
    """
    Computes the cumulative max of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, -1, 2, -2, 3, -3])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cummax(0).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.MAX)

  def triu(self, diagonal:int=0) -> Tensor:
    """
    Returns the upper triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=-1).numpy())
    ```
    """
    return Tensor._tri(self.shape[-2], self.shape[-1], diagonal=diagonal, device=self.device, dtype=dtypes.bool).where(self, 0).cast(self.dtype)

  def tril(self, diagonal:int=0) -> Tensor:
    """
    Returns the lower triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=-1).numpy())
    ```
    """
    return Tensor._tri(self.shape[-2], self.shape[-1], diagonal=diagonal+1, device=self.device, dtype=dtypes.bool).where(0, self).cast(self.dtype)


  # ***** unary ops *****

  def logical_not(self):
    """
    Computes the logical NOT of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([False, True]).logical_not().numpy())
    ```
    """
    return F.Neq.apply(*self.cast(dtypes.bool)._broadcasted(True))
  def neg(self):
    """
    Negates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
    ```
    """
    return self*-1 if self.dtype != dtypes.bool else self.logical_not()
  def contiguous(self):
    """
    Returns a contiguous tensor.
    """
    return F.Contiguous.apply(self)
  def contiguous_backward(self):
    """
    Inserts a contiguous operation in the backward pass.
    """
    return F.ContiguousBackward.apply(self)
  def log(self):
    """
    Computes the natural logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log().numpy())
    ```
    """
    return F.Log.apply(self.cast(least_upper_float(self.dtype)))
  def log2(self):
    """
    Computes the base-2 logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log2().numpy())
    ```
    """
    return self.log()/math.log(2)
  def exp(self):
    """
    Computes the exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp().numpy())
    ```
    """
    return F.Exp.apply(self.cast(least_upper_float(self.dtype)))
  def exp2(self):
    """
    Computes the base-2 exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp2().numpy())
    ```
    """
    return F.Exp.apply(self*math.log(2))
  def relu(self):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    - Described: https://paperswithcode.com/method/relu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
    ```
    """
    return F.Relu.apply(self)
  def sigmoid(self):
    """
    Applies the Sigmoid function element-wise.

    - Described: https://en.wikipedia.org/wiki/Sigmoid_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
    ```
    """
    return F.Sigmoid.apply(self.cast(least_upper_float(self.dtype)))
  def hardsigmoid(self, alpha:float=1/6, beta:float=0.5):
    """
    Applies the Hardsigmoid function element-wise.
    NOTE: default `alpha` and `beta` values is taken from torch

    - Described: https://paperswithcode.com/method/hard-sigmoid
    - See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardsigmoid().numpy())
    ```
    """
    return (alpha * self + beta).relu() - (alpha * self + beta - 1).relu()

  def sqrt(self):
    """
    Computes the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).sqrt().numpy())
    ```
    """
    return F.Sqrt.apply(self.cast(least_upper_float(self.dtype)))
  def rsqrt(self):
    """
    Computes the reciprocal of the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
    ```
    """
    return self.reciprocal().sqrt()
  def sin(self):
    """
    Computes the sine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
    ```
    """
    return F.Sin.apply(self.cast(least_upper_float(self.dtype)))
  def cos(self):
    """
    Computes the cosine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
    ```
    """
    return ((math.pi/2)-self).sin()
  def tan(self):
    """
    Computes the tangent of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
    ```
    """
    return self.sin() / self.cos()

  def asin(self):
    """
    Computes the inverse sine (arcsine) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).asin().numpy())
    ```
    """
    # https://personal.math.ubc.ca/~cbm/aands/page_81.htm 4.4.46
    coefficients = [-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810, -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050]
    x = math.pi / 2 - (1.0 - self.abs()).sqrt() * polyN(self.abs(), coefficients)
    return self.sign() * x

  def acos(self):
    """
    Computes the inverse cosine (arccosine) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).acos().numpy())
    ```
    """
    return math.pi / 2 - self.asin()

  def atan(self):
    """
    Computes the inverse tangent (arctan) of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).atan().numpy())
    ```
    """
    return (self / (1 + self * self).sqrt()).asin()

  # ***** math functions *****

  def trunc(self: Tensor) -> Tensor:
    """
    Truncates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).trunc().numpy())
    ```
    """
    return self.cast(dtypes.int32).cast(self.dtype)
  def ceil(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise towards positive infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).ceil().numpy())
    ```
    """
    return (self > (b := self.trunc())).where(b+1, b)
  def floor(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise towards negative infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).floor().numpy())
    ```
    """
    return (self < (b := self.trunc())).where(b-1, b)
  def round(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise with rounding half to even.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).round().numpy())
    ```
    """
    return ((self > 0) == ((b := self.cast(dtypes.int32) / 2.0).cast(dtypes.int32) == b)).where((self - 0.5).ceil(), (self + 0.5).floor())

  def isinf(self:Tensor, detect_positive:bool=True, detect_negative:bool=True):
    """
    Checks the tensor element-wise to return True where the element is infinity, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf().numpy())
    ```
    """
    return (self == float("inf")) * detect_positive + (self == float("-inf")) * detect_negative
  def isnan(self:Tensor):
    """
    Checks the tensor element-wise to return True where the element is NaN, otherwise returns False

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan().numpy())
    ```
    """
    return self != self

  def lerp(self, end: Tensor, weight: Union[Tensor, float]) -> Tensor:
    """
    Linearly interpolates between `self` and `end` by `weight`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
    ```
    """
    if self.dtype == dtypes.uint8 and isinstance(weight, Tensor):
      w_i = (weight * (1<<(W_PREC:=7)) + 0.5).cast(dtypes.int16)
      return (self+(((end - self).cast(dtypes.int8) * w_i + (1<<W_PREC-1)).cast(dtypes.uint16) >> W_PREC)).cast(dtypes.uint8)
    return self + (end - self) * weight

  def square(self):
    """
    Squares the tensor element-wise.
    Equivalent to `self*self`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
    ```
    """
    return self*self
  def clamp(self, min_=None, max_=None):
    """
    Clips (clamps) the values in the tensor between `min_` and `max_` element-wise.
    If `min_` is `None`, there is no lower bound. If `max_` is None, there is no upper bound.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).clip(-1, 1).numpy())
    ```
    """
    if min_ is None and max_ is None: raise RuntimeError("at least one of 'min_' or 'max_' must not be None")
    ret = self.maximum(min_) if min_ is not None else self
    return ret.minimum(max_) if max_ is not None else ret
  def clip(self, min_=None, max_=None):
    """
    Alias for `Tensor.clamp`.
    """
    return self.clamp(min_, max_)
  def sign(self):
    """
    Returns the sign of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
    ```
    """
    return F.Sign.apply(self)
  def abs(self):
    """
    Computes the absolute value of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
    ```
    """
    return self * self.sign()
  def reciprocal(self):
    """
    Compute `1/x` element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
    ```
    """
    return F.Reciprocal.apply(self.cast(least_upper_float(self.dtype)))

  # ***** activation functions *****

  def elu(self, alpha=1.0):
    """
    Applies the Exponential Linear Unit (ELU) function element-wise.

    - Described: https://paperswithcode.com/method/elu
    - Paper: https://arxiv.org/abs/1511.07289v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
    ```
    """
    return self.relu() - alpha*(1-self.exp()).relu()

  def celu(self, alpha=1.0):
    """
    Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

    - Described: https://paperswithcode.com/method/celu
    - Paper: https://arxiv.org/abs/1704.07483

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
    ```
    """
    return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)

  def selu(self, alpha=1.67326, gamma=1.0507):
    """
    Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

    - Described: https://paperswithcode.com/method/selu
    - Paper: https://arxiv.org/abs/1706.02515v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).selu().numpy())
    ```
    """
    return gamma * (self >= 0).detach().where(self, alpha * (self.exp() - 1))

  def swish(self):
    """
    See `.silu()`

    - Paper: https://arxiv.org/abs/1710.05941v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
    ```
    """
    return self * self.sigmoid()

  def silu(self):
    """
    Applies the Sigmoid Linear Unit (SiLU) function element-wise.

    - Described: https://paperswithcode.com/method/silu
    - Paper: https://arxiv.org/abs/1606.08415

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
    ```
    """
    return self.swish()   # The SiLU function is also known as the swish function.

  def relu6(self):
    """
    Applies the ReLU6 function element-wise.

    - Described: https://paperswithcode.com/method/relu6
    - Paper: https://arxiv.org/abs/1704.04861v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
    ```
    """
    return self.relu() - (self-6).relu()

  def hardswish(self):
    """
    Applies the Hardswish function element-wise.

    - Described: https://paperswithcode.com/method/hard-swish
    - Paper: https://arxiv.org/abs/1905.02244v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
    ```
    """
    return self * (self+3).relu6() * (1/6)

  def tanh(self):
    """
    Applies the Hyperbolic Tangent (tanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
    ```
    """
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def sinh(self):
    """
    Applies the Hyperbolic Sine (sinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
    ```
    """
    return (self.exp() - self.neg().exp()) / 2

  def cosh(self):
    """
    Applies the Hyperbolic Cosine (cosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
    ```
    """
    return (self.exp() + self.neg().exp()) / 2

  def atanh(self):
    """
    Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
    ```
    """
    return ((1 + self)/(1 - self)).log() / 2

  def asinh(self):
    """
    Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
    ```
    """
    return (self + (self.square() + 1).sqrt()).log()

  def acosh(self):
    """
    Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
    ```
    """
    return (self + (self.square() - 1).sqrt()).log()

  def hardtanh(self, min_val=-1, max_val=1):
    """
    Applies the Hardtanh function element-wise.

    - Described: https://paperswithcode.com/method/hardtanh-activation

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
    ```
    """
    return self.clip(min_val, max_val)

  def erf(self):
    """
    Applies error function element-wise.

    - Described: https://en.wikipedia.org/wiki/Error_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).erf().numpy())
    ```
    """
    # https://personal.math.ubc.ca/~cbm/aands/page_299.htm 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * self.abs())
    return self.sign() * (1.0 - t * polyN(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592]) * (-self.square()).exp())

  def gelu(self):
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    - Described: https://paperswithcode.com/method/gelu
    - Paper: https://arxiv.org/abs/1606.08415v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
    ```
    """
    return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self ** 3)).tanh())

  def quick_gelu(self):
    """
    Applies the Sigmoid GELU approximation element-wise.

    - Described: https://paperswithcode.com/method/gelu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
    ```
    """
    return self * (self * 1.702).sigmoid()

  def leakyrelu(self, neg_slope=0.01):
    """
    Applies the Leaky ReLU function element-wise.

    - Described: https://paperswithcode.com/method/leaky-relu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leakyrelu().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leakyrelu(neg_slope=0.42).numpy())
    ```
    """
    return self.relu() - (-neg_slope*self).relu()

  def mish(self):
    """
    Applies the Mish function element-wise.

    - Described: https://paperswithcode.com/method/mish
    - Paper: https://arxiv.org/abs/1908.08681v3

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
    ```
    """
    return self * self.softplus().tanh()

  def softplus(self, beta=1):
    """
    Applies the Softplus function element-wise.

    - Described: https://paperswithcode.com/method/softplus

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
    ```
    """
    return (1/beta) * (1 + (self*beta).exp()).log()

  def softsign(self):
    """
    Applies the Softsign function element-wise.

    - Described: https://paperswithcode.com/method/softsign

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
    ```
    """
    return self / (1 + self.abs())

  # ***** broadcasted elementwise ops *****
  def _broadcast_to(self, new_shape:Tuple[sint, ...]) -> Tensor:
    if self.shape == new_shape: return self
    if self.ndim > len(new_shape): raise ValueError(f"cannot broadcast tensor to fewer dimensions. shape={self.shape} to {new_shape=}")
    # first unsqueeze left with 1s https://data-apis.org/array-api/latest/API_specification/broadcasting.html
    shape, _ = _align_left(self.shape, new_shape)
    # for each dimension, check either dim is 1, or it does not change
    if not all(resolve(s == ns) or resolve(s == 1) for s,ns in zip(shape, new_shape)):
      raise ValueError(f"cannot broadcast {self.shape} to {new_shape=}")
    return F.Expand.apply(self.reshape(shape), shape=new_shape)

  def _broadcasted(self, y:Union[Tensor, UOp, ConstType], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      # make y a Tensor
      assert isinstance(y, (*get_args(ConstType), UOp)), f"{type(y)=}, {y=}"
      if isinstance(x.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype
      elif not isinstance(y, UOp): y_dtype = dtypes.from_py(y)
      if isinstance(y, UOp): y = Tensor.from_uop(y, device=x.device)
      else: y = Tensor(dtypes.as_const(y, y_dtype), x.device, y_dtype, requires_grad=False)

    if match_dtype and x.dtype != y.dtype:
      output_dtype = least_upper_dtype(x.dtype, y.dtype)
      x, y = x.cast(output_dtype), y.cast(output_dtype)

    if reverse: x, y = y, x

    # broadcast
    return x._broadcast_to(out_shape:=_broadcast_shape(x.shape, y.shape)), y._broadcast_to(out_shape)

  def _to_const_val(self, x:Union[Tensor, ConstType]) -> Union[Tensor, ConstType]:
    return x.lazydata.base.arg if isinstance(x, Tensor) and isinstance(x.lazydata, LazyBuffer) and x.lazydata.is_unrealized_unmasked_const() \
      and not x.requires_grad and self._broadcasted(x)[0].shape == self.shape else x

  def add(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Adds `self` and `x`.
    Equivalent to `self + x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.add(20).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.add(Tensor([[2.0], [3.5]])).numpy())
    ```
    """
    return F.Add.apply(*self._broadcasted(x, reverse))

  def sub(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Subtracts `x` from `self`.
    Equivalent to `self - x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sub(20).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sub(Tensor([[2.0], [3.5]])).numpy())
    ```
    """
    a, b = self._broadcasted(x, reverse)
    return a + (-b)

  def mul(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Multiplies `self` and `x`.
    Equivalent to `self * x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mul(3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mul(Tensor([[-1.0], [2.0]])).numpy())
    ```
    """
    return F.Mul.apply(*self._broadcasted(x, reverse))

  def idiv(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Divides `self` by `x`.
    Equivalent to `self // x`.
    Supports broadcasting to a common shape, type promotion, and integer inputs.
    `idiv` performs integer division.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 4, 10]).idiv(Tensor([2, 3, 4])).numpy())
    ```
    """
    return F.IDiv.apply(*self._broadcasted(x, reverse))

  def div(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Divides `self` by `x`.
    Equivalent to `self / x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.
    `div` performs true division.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.div(3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
    ```
    """
    numerator, denominator = self._broadcasted(x, reverse)
    return numerator.cast(least_upper_float(numerator.dtype)) * denominator.cast(least_upper_float(denominator.dtype)).reciprocal()

  def xor(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Computes bitwise xor of `self` and `x`.
    Equivalent to `self ^ x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, -2, 3]).xor(Tensor([1, 0, 3])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).xor(Tensor([True, False, True, False])).numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return F.Xor.apply(*self._broadcasted(x, reverse))

  def bitwise_and(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Compute the bit-wise AND of `self` and `x`.
    Equivalent to `self & x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).bitwise_and(Tensor([True, False, True, False])).numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return F.BitwiseAnd.apply(*self._broadcasted(x, reverse))

  def bitwise_or(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Compute the bit-wise OR of `self` and `x`.
    Equivalent to `self | x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).bitwise_or(Tensor([True, False, True, False])).numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return F.BitwiseOr.apply(*self._broadcasted(x, reverse))

  def bitwise_not(self) -> Tensor:
    """
    Compute the bit-wise NOT of `self`.
    Equivalent to `~self`.
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, False]).bitwise_not().numpy())
    ```
    """
    if self.dtype != dtypes.bool and not dtypes.is_int(self.dtype): raise RuntimeError(f"{self.dtype} is not supported")
    return self.logical_not() if self.dtype == dtypes.bool else self ^ ((1<<8*self.dtype.itemsize)-1)

  def lshift(self, x:int):
    """
    Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self << x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0, f"not supported {self.dtype=} {x=}"
    return self.mul(2 ** x)

  def rshift(self, x:int):
    """
    Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self >> x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0, f"not supported {self.dtype=} {x=}"
    return self.idiv(2 ** x)

  def pow(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Computes power of `self` with `x`.
    Equivalent to `self ** x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(Tensor([-1.5, 0.5, 1.5])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((2 ** Tensor([-1, 2, 3])).numpy())
    ```
    """
    x = self._to_const_val(x)
    if not isinstance(x, Tensor) and not reverse:
      # simple pow identities
      if x < 0: return self.reciprocal().pow(-x)
      if x == 0: return 1 + self * 0
      if int(x - 0.5) + 0.5 == x: return self.pow(int(x - 0.5)) * self.sqrt()
      if int(x) == x: return self.pow(x // 2).square() * (1 if x % 2 == 0 else self)

    # positive const ** self
    if not isinstance(x, Tensor) and reverse and x > 0: return self.mul(math.log(x)).exp()

    base, exponent = self._broadcasted(x, reverse=reverse)
    # start with b ** e = exp(e * log(b))
    ret = base.abs().log().mul(exponent).exp()
    # correct sign of negative base with odd exponent (cos has a period of 2pi so we use it here to get the oddness of the exponent)
    negative_base = (base < 0).detach().where(1, 0)
    # 1 for non-negative base or negative even exponent, -1 for negative odd exponent, don't care about non-integer exponent
    correct_sign = 1 + negative_base * ((exponent * math.pi).cos() - 1)
    # inject nan for negative base and non-integer exponent
    inject_nan = (negative_base * (exponent != exponent.trunc())).detach().where(math.nan, 1)
    # apply correct_sign inject_nan, and fix 0 ** 0 = 1
    return ((base == 0) * (exponent == 0)).detach().where(1, ret * correct_sign * inject_nan)

  def maximum(self, x:Union[Tensor, ConstType]) -> Tensor:
    """
    Computes element-wise maximum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return (self<x).detach().where(x, (self==x).detach().where(((self * 0.5 + x * 0.5).cast(self.dtype)), self))

  def minimum(self, x:Union[Tensor, ConstType]) -> Tensor:
    """
    Computes element-wise minimum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return -((-self).maximum(-x))

  def where(self:Tensor, x:Union[Tensor, ConstType, sint], y:Union[Tensor, ConstType, sint]):
    """
    Return a tensor of elements selected from either `x` or `y`, depending on `self`.
    `output_i = x_i if self_i else y_i`.

    ```python exec="true" source="above" session="tensor" result="python"
    cond = Tensor([[True, True, False], [True, False, False]])
    print(cond.where(1, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    cond = Tensor.randn(2, 3)
    print(cond.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((cond > 0).where(cond, -float("inf")).numpy())
    ```
    """
    if isinstance(x, Tensor): x, y = x._broadcasted(y)
    elif isinstance(y, Tensor): y, x = y._broadcasted(x)
    cond, x = self._broadcasted(x, match_dtype=False)
    cond, y = cond._broadcasted(y, match_dtype=False)
    return F.Where.apply(cond.cast(dtypes.bool), *x._broadcasted(y))

  def masked_fill(self:Tensor, mask:Tensor, value:Union[Tensor, ConstType]): return mask.where(value, self)

  # ***** op wrappers *****

  def __invert__(self) -> Tensor: return self.bitwise_not()

  def __lshift__(self, x) -> Tensor: return self.lshift(x)
  def __rshift__(self, x) -> Tensor: return self.rshift(x)

  def __pow__(self, x) -> Tensor: return self.pow(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)

  def __rpow__(self, x) -> Tensor: return self.pow(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __ifloordiv__(self, x) -> Tensor: return self.assign(self.idiv(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))
  def __iand__(self, x) -> Tensor: return self.assign(self.bitwise_and(x))
  def __ior__(self, x) -> Tensor: return self.assign(self.bitwise_or(x))
  def __ixor__(self, x) -> Tensor: return self.assign(self.xor(x))
  def __ilshift__(self, x) -> Tensor: return self.assign(self.lshift(x))
  def __irshift__(self, x) -> Tensor: return self.assign(self.rshift(x))

  def __lt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, False))
  def __gt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, True))
  def ne(self, x) -> Tensor: return F.Neq.apply(*self._broadcasted(x))

  def __eq__(self, x) -> Tensor: return self.eq(x)                      # type: ignore[override]

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    """
    Applies a linear transformation to `self` using `weight` and `bias`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    weight = Tensor([[1, 2], [3, 4]])
    bias = Tensor([1, 2])
    print(t.linear(weight, bias).numpy())
    ```
    """
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]):
    """
    Applies a sequence of functions to `self` chaining the output of each function to the input of the next.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
    ```
    """
    return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis:Union[int,Tuple[int,...]]=-1, eps:float=1e-5) -> Tensor:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    - Described: https://paperswithcode.com/method/layer-normalization
    - Paper: https://arxiv.org/abs/1607.06450v1

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 10, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.layernorm()
    print(t.mean().item(), t.std().item())
    ```
    """
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor, axis:Union[int,Tuple[int,...]]=1) -> Tensor:
    """
    Applies Batch Normalization over a mini-batch of inputs.

    - Described: https://paperswithcode.com/method/batch-normalization
    - Paper: https://arxiv.org/abs/1502.03167

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 4, 16, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.batchnorm(None, None, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
    print(t.mean().item(), t.std().item())
    ```
    """
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight is not None: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return (ret + bias.reshape(shape)) if bias is not None else ret

  def dropout(self, p=0.5) -> Tensor:
    """
    Applies dropout to `self`.

    NOTE: dropout is only applied when `Tensor.training` is `True`.

    - Described: https://paperswithcode.com/method/dropout
    - Paper: https://jmlr.org/papers/v15/srivastava14a.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 2)
    with Tensor.train():
      print(t.dropout().numpy())
    ```
    """
    if not Tensor.training or p == 0: return self
    return (Tensor.rand_like(self, requires_grad=False, dtype=dtypes.default_float, contiguous=False) >= p).contiguous().where(self, 0) / (1.0 - p)

  # helper function commonly used for indexing
  def _one_hot_along_dim(self:Tensor, num_classes:sint, dim:int=-1):
    offset = self.ndim - self._resolve_dim(dim) - 1
    return self == Tensor.arange(num_classes, device=self.device, requires_grad=False).reshape((num_classes,) + (1,) * offset)

  def one_hot(self, num_classes:int=-1) -> Tensor:
    """
    Converts `self` to a one-hot tensor.

    `num_classes` defaults to -1, which means num_classes will be inferred as max(self) + 1.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, 3, 3, 4])
    print(t.one_hot(5).numpy())
    ```
    """
    if num_classes == -1: num_classes = (self.max()+1).item()
    return self[..., None]._one_hot_along_dim(num_classes).where(1, 0)

  def sparse_categorical_crossentropy(self, Y:Tensor, ignore_index:int=-1, label_smoothing=0.0, reduction:ReductionStr="mean") -> Tensor:
    """
    Computes the sparse categorical cross-entropy loss between `self` and `Y`.

    NOTE: `self` is logits and `Y` is the target labels.
    NOTE: unlike PyTorch, this function expects the class axis to be -1

    See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.sparse_categorical_crossentropy(Y).item())
    ```
    """
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    assert reduction in ("mean", "sum", "none"), "reduction must be one of ['mean', 'sum', 'none']"
    log_probs, loss_mask = self.log_softmax(), (Y != ignore_index) if ignore_index != -1 else Y.ones_like(dtype=dtypes.bool)
    y_counted = Y.to(self.device).flatten().reshape(-1, 1)._one_hot_along_dim(self.shape[-1])
    y = (y_counted * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
    unreduced = ((1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing)
    # NOTE: because of ignore_index, we can't use Tensor.mean (so can't use `_do_reduction` here)
    return -(unreduced.sum() / loss_mask.sum() if reduction == "mean" else (unreduced.sum() if reduction == "sum" else unreduced))

  # ***** Tensor Properties *****

  @property
  def ndim(self) -> int:
    """
    Returns the number of dimensions in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.ndim)
    ```
    """
    return len(self.shape)

  def numel(self) -> sint:
    """
    Returns the total number of elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(t.numel())
    ```
    """
    return prod(self.shape)

  def element_size(self) -> int:
    """
    Returns the size in bytes of an individual element in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([5], dtype=dtypes.int16)
    print(t.element_size())
    ```
    """
    return self.dtype.itemsize

  def nbytes(self) -> int:
    """
    Returns the total number of bytes of all elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float)
    print(t.nbytes())
    ```
    """
    return self.numel() * self.element_size()

  def is_floating_point(self) -> bool:
    """
    Returns `True` if the tensor contains floating point types, i.e. is one of `dtype.float64`, `dtype.float32`,
    `dtype.float16`, `dtype.bfloat16`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float32)
    print(t.is_floating_point())
    ```
    """
    return dtypes.is_float(self.dtype)

  def size(self, dim:Optional[int]=None) -> Union[sint, Tuple[sint, ...]]:
    """
    Return the size of the tensor. If `dim` is specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[4, 5, 6], [7, 8, 9]])
    print(t.size())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.size(dim=1))
    ```
    """
    return self.shape if dim is None else self.shape[dim]

  # ***** cast ops *****

  def llvm_bf16_cast(self, dtype:DTypeLike):
    # hack for devices that don't support bfloat16
    assert self.dtype == dtypes.bfloat16
    return self.to("LLVM").bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).cast(dtype)

  def cast(self, dtype:DTypeLike) -> Tensor:
    """
    Casts `self` to the given `dtype`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2.5, 3], dtype=dtypes.float)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.cast(dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    """
    return self if self.dtype == (dt:=to_dtype(dtype)) else F.Cast.apply(self, dtype=dt)

  def bitcast(self, dtype:DTypeLike) -> Tensor:
    """
    Bitcasts `self` to the given `dtype` of the same itemsize.

    `self` must not require a gradient.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bitcast(dtypes.uint32)
    print(t.dtype, t.numpy())
    ```
    """
    if self.requires_grad: raise RuntimeError("can't backprop through bitcast")
    dt = to_dtype(dtype)
    if (not isinstance(self.device, str) or not self.device.startswith("DISK")) and (ns:=dt.itemsize) != (os:=self.dtype.itemsize):
      if (self.shape[-1]*os) % ns != 0: raise RuntimeError("unsupported size in bitcast")
      new_uint, old_uint = to_dtype(f"uint{8*ns}"), to_dtype(f"uint{8*os}")
      tmp = self.bitcast(old_uint)
      if ns > os: return functools.reduce(Tensor.add, (tmp[..., i::ns//os].cast(new_uint) << 8*i*os for i in range(ns//os))).bitcast(dtype)
      return Tensor.stack(*(tmp>>8*i*ns for i in range(os//ns)), dim=-1).flatten(-2).cast(new_uint).bitcast(dtype)
    return F.Cast.apply(self, dtype=dt, bitcast=True) if self.dtype != dt else self

  def float(self) -> Tensor:
    """
    Convenience method to cast `self` to a `float32` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.float()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.float32)

  def half(self) -> Tensor:
    """
    Convenience method to cast `self` to a `float16` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.half()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.float16)

  def int(self) -> Tensor:
    """
    Convenience method to cast `self` to a `int32` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.int()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.int32)

  def bool(self) -> Tensor:
    """
    Convenience method to cast `self` to a `bool` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 0, 1])
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bool()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.bool)

  # *** image Tensor function replacements ***

def _metadata_wrapper(fn):
  def _wrapper(*args, **kwargs):
    if _METADATA.get() is not None: return fn(*args, **kwargs)

    caller = ""

    token = _METADATA.set(Metadata(name=fn.__name__, caller=caller))
    ret = fn(*args, **kwargs)
    _METADATA.reset(token)
    return ret
  return _wrapper

if TRACEMETA >= 1:
  for name, fn in inspect.getmembers(Tensor, inspect.isfunction):
    if name in ["__class__", "__init__", "__new__", "__repr__", "backward", "sequential"]: continue
    setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))
