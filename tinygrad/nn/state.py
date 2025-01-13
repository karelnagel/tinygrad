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
  """
  Loads a .safetensor file from disk, returning the data, metadata length, and metadata.
  """
  data_start = int.from_bytes(t[0:8].data(), "little") + 8
  return t, data_start, json.loads(t[8:data_start].data().tobytes())
def safe_load(fn:Union[Tensor, str, pathlib.Path]) -> Dict[str, Tensor]:
  """
  Loads a .safetensor file from disk, returning the state_dict.
  ```python
  state_dict = nn.state.safe_load("test.safetensor")
  ```
  """
  t, data_start, metadata = safe_load_metadata(fn)
  data = t[data_start:]
  return { k: data[v['data_offsets'][0]:v['data_offsets'][1]].bitcast(safe_dtypes[v['dtype']]).reshape(v['shape'])
          for k, v in metadata.items() if k != "__metadata__" }

def safe_save(tensors:Dict[str, Tensor], fn:str, metadata:Optional[Dict[str, Any]]=None):
  """
  Saves a state_dict to disk in a .safetensor file with optional metadata.
  ```python
  t = Tensor([1, 2, 3])
  nn.state.safe_save({'t':t}, "test.safetensor")
  ```
  """
  headers, offset = {}, 0
  if metadata: headers['__metadata__'] = metadata
  for k,v in tensors.items():
    headers[k] = {'dtype': inverse_safe_dtypes[v.dtype], 'shape': list(v.shape), 'data_offsets':[offset, offset+v.nbytes()]}
    offset += v.nbytes()
  j = json.dumps(headers, separators=(',', ':'))
  j += "\x20"*((8-len(j)%8)%8)
  pathlib.Path(fn).unlink(missing_ok=True)
  t = Tensor.empty(8+len(j)+offset, dtype=dtypes.uint8, device=f"disk:{fn}")
  t[0:8].bitcast(dtypes.int64).assign([len(j)])
  t[8:8+len(j)].assign(list(j.encode('utf-8')))
  for k,v in safe_load(t).items(): v.assign(tensors[k])
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
  """
  Loads a state_dict into a model.
  ```python
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)
  net = Net()
  state_dict = nn.state.get_state_dict(net)
  nn.state.load_state_dict(net, state_dict)
  ```
  """
  start_mem_used = GlobalCounters.mem_used
  with Timing("loaded weights in ", lambda et_ns: f", {(B:=(GlobalCounters.mem_used-start_mem_used))/1e9:.2f} GB loaded at {B/et_ns:.2f} GB/s"):
    model_state_dict = get_state_dict(model)
    if DEBUG >= 1 and len(state_dict) > len(model_state_dict):
      print("WARNING: unused weights in state_dict", sorted(list(state_dict.keys() - model_state_dict.keys())))
    for k,v in (t := tqdm(model_state_dict.items(), disable=CI or not verbose)):
      t.desc = f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}: "
      if k not in state_dict and not strict:
        if DEBUG >= 1: print(f"WARNING: not loading {k}")
        continue
      if v.shape != state_dict[k].shape:
        raise ValueError(f'Shape mismatch in layer `{k}`: Expected shape {v.shape}, but found {state_dict[k].shape} in state dict.')
      # if isinstance((mlb:=v.lazydata), MultiLazyBuffer):
      #   if isinstance(state_dict[k].lazydata, MultiLazyBuffer): v.replace(state_dict[k]).realize()
      #   else: v.replace(state_dict[k].shard(mlb.device, mlb.axis)).realize()
      else: v.replace(state_dict[k].to(v.device)).realize()
      if consume: del state_dict[k]

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