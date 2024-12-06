import os, json, pathlib, zipfile, pickle, tarfile, struct, functools, io
from typing import Dict, Union, List, Optional, Any, Tuple, Callable, BinaryIO, Iterable, TypeVar
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, GlobalCounters, tqdm
from tinygrad.shape.view import strides_for_shape

class TensorIO(io.RawIOBase, BinaryIO):
  pass
safe_dtypes = {"BOOL":dtypes.bool, "I8":dtypes.int8, "U8":dtypes.uint8, "I16":dtypes.int16, "U16":dtypes.uint16, "I32":dtypes.int, "U32":dtypes.uint,
               "I64":dtypes.int64, "U64":dtypes.uint64, "F16":dtypes.float16, "BF16":dtypes.bfloat16, "F32":dtypes.float32, "F64":dtypes.float64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

R = TypeVar('R')
def accept_filename(func: Callable[[Tensor], R]) -> Callable[[Union[Tensor, str, pathlib.Path]], R]:
  @functools.wraps(func)
  def wrapper(fn: Union[Tensor, str, pathlib.Path]) -> R: return func(Tensor(pathlib.Path(fn)) if not isinstance(fn, Tensor) else fn)
  return wrapper

@accept_filename
def safe_load_metadata(t:Tensor) -> Tuple[Tensor, int, Dict[str, Any]]:
  pass
def safe_load(fn:Union[Tensor, str, pathlib.Path]) -> Dict[str, Tensor]:
  pass
def safe_save(tensors:Dict[str, Tensor], fn:str, metadata:Optional[Dict[str, Any]]=None):
  pass
# state dict

from collections import OrderedDict
def get_state_dict(obj, prefix:str='', tensor_type=Tensor) -> Dict[str, Tensor]:
  """
  Returns a state_dict of the object, with optional prefix.

  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(nn.state.get_state_dict(net).keys())
  ```
  """
  if isinstance(obj, tensor_type): return {prefix.strip('.'):obj}
  if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  # namedtuple
  if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
  if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
  state_dict = {}
  if isinstance(obj, (list, tuple)):
    for i,x in enumerate(obj): state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
  elif isinstance(obj, dict):
    for k,v in obj.items(): state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))
  return state_dict
def get_parameters(obj) -> List[Tensor]:
  """
  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(len(nn.state.get_parameters(net)))
  ```
  """
  return list(get_state_dict(obj).values())

def load_state_dict(model, state_dict:Dict[str, Tensor], strict=True, verbose=True, consume=False) -> None:
  pass
@accept_filename
def tar_extract(t: Tensor) -> Dict[str, Tensor]:
  pass
# torch support!

def torch_load(fn:str) -> Dict[str, Tensor]:
  pass
def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
  pass
@accept_filename
def gguf_load(tensor: Tensor) -> Tuple[Dict, Dict[str, Tensor]]:
  pass