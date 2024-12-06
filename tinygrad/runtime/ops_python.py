# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
import sys
from typing import Tuple, List, Optional, Any, Dict, TYPE_CHECKING
import pickle, base64, itertools, time, struct
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, flatten
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.ops import exec_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer
# from tinygrad.renderer.cstyle import CUDARenderer, MetalRenderer, AMDRenderer, IntelRenderer, ClangRenderer

def _load(m, i):pass

def load(inp, j=0):pass

def _store(m, i, v):pass

class PythonProgram: pass

class PythonRenderer(Renderer):
  device = "PYTHON"
  def __init__(self):pass

  def render(self, name:str, uops:List[UOp]) -> str: pass

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, PythonAllocator(), PythonRenderer(), PythonCompiler(), PythonProgram)
