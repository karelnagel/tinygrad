# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Tuple, List, Optional, Any, Dict
import pickle, base64, itertools, time, struct
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, flatten
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.ops import exec_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer

def _load(m, i):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]

def load(inp, j=0):
  if len(inp) == 3: return [_load(m, x+j if x is not None else None) if gate else default for (m,x),default,gate in zip(*inp)]
  return [_load(m, x+j if x is not None else None) for m,x in inp[0]]

def _store(m, i, v):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

class PythonProgram: pass

class PythonRenderer(Renderer):
  device = "PYTHON"
  def __init__(self): pass

  def render(self, name:str, uops:List[UOp]) -> str:
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, PythonAllocator(), PythonRenderer(), PythonCompiler(), PythonProgram)
