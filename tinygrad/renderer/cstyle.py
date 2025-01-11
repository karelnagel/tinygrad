from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, DefaultDict, Literal, Callable, cast
import os, math
from collections import defaultdict, Counter
from tinygrad.ops import GroupOp, Ops, UOp, PatternMatcher, UPat, cast_float_to_bf16
from tinygrad.helpers import strip_parens, getenv, prod, dedup, AMX
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.renderer import Renderer, TensorCore

base_rewrite = PatternMatcher([
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{ctx[x.src[0]]} = {ctx[x.src[1]]};"),
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"if ({ctx[x.src[0]]}) {{"),
  (UPat((Ops.ENDIF, Ops.ENDRANGE)), lambda ctx: "}"),
  (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
  # r method accesses
  (UPat(Ops.RANGE, name="x"),
   lambda ctx,x: f"for ({ctx.render_dtype(x.dtype)} {ctx[x]} = {ctx[x.src[0]]}; {ctx[x]} < {ctx[x.src[1]]}; {ctx[x]}++) {{"),
  (UPat(Ops.VECTORIZE, name="x"),
   lambda ctx,x: f"{ctx.float4.replace('float4', ctx.render_dtype(x.dtype))}" + \
    (f"{{{','.join([ctx[y] for y in x.src])}}}" if ctx.device == "CLANG" else f"({','.join([ctx[y] for y in x.src])})")),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx[x.src[0]])})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"(*(({ctx.buffer_prefix}{ctx.render_dtype(x.dtype)}*)&{ctx[x.src[0]]}))"),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"{ctx.smem_align}{ctx.smem_prefix}{ctx.render_dtype(x.dtype.base)} {ctx[x]}[{x.arg[1]}];"),
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.NOOP, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0][0]](x.arg[0][-1])}; /* {x.arg[1]} */"),
  # const
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, ctx.infinity)})"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, f'-{ctx.infinity}')})"),
  (UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx.nan)})" if math.isnan(x.arg) else None),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{float(x.arg)}f"),
  (UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}ll"),
  (UPat(Ops.CONST, dtype=dtypes.uint64, name="x"), lambda ctx,x: f"{x.arg}ull"),
  (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), lambda ctx,x: f"{x.arg}u"),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "1" if x.arg else "0"),
  # consts are rendered to larger type and casted
  (UPat(Ops.CONST, (dtypes.bfloat16, dtypes.half), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}f')})"),
  (UPat(Ops.CONST, (dtypes.uint8, dtypes.uint16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}u')})"),
  (UPat(Ops.CONST, (dtypes.int8, dtypes.int16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, x.arg)})"),
  # default const render
  (UPat(Ops.CONST, name="x"), lambda ctx,x: str(x.arg)),
  # new load/store
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx'))),
   lambda ctx,buf,idx: f"({ctx[buf]}+{strip_parens(ctx[idx]) if idx.arg == Ops.ADD else ctx[idx]})"),
  (UPat(Ops.LOAD, src=(UPat.var('bidx'), UPat.var("var"), UPat.var("gate"))), lambda ctx,bidx,var,gate: f"({ctx[gate]}?*{ctx[bidx]}:{ctx[var]})"),
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx,bidx: f"*{ctx[bidx]}"),
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx,bidx,var: f"*{ctx[bidx]} = {ctx[var]};"),
  # alu/gep
  (UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
    *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR} else ctx[v] for v in x.src]), x.dtype)),
  (UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + \
    (f"[{x.arg[0]}]" if x.src[0].dtype.count > (8 if ctx.device in {"CUDA", "NV"} else 4) or ctx.device == 'CLANG' else f".{'xyzwabcd'[x.arg[0]]}")),
])

extra_pm = PatternMatcher([
  # insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
  (UPat(Ops.BITCAST, name="x"),
   lambda x: UOp(Ops.BITCAST, x.dtype, (UOp(Ops.NOOP, x.src[0].dtype, x.src),)) if x.src[0].op is not Ops.NOOP else None),
  # rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

def uops_to_dtypes(uops:List[UOp]) -> List[DType]: return dedup(u.dtype for u in uops if not isinstance(u.dtype, (ImageDType, PtrDType)))

class CStyleLanguage(Renderer):
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}
  extra_args: List[str] = []
  float4: Optional[str] = None
  type_map: Dict[DType, str] = {}
  infinity: str = "INFINITY"
  nan: str = "NAN"
  code_for_op: Dict = {
    Ops.SQRT: lambda x,dtype: f"sqrt({x})", Ops.RECIP: lambda x,dtype: f"(1/{x})", Ops.NEG: lambda x,dtype: f"-{x}",
    Ops.EXP2: lambda x,dtype: f"exp2({x})", Ops.LOG2: lambda x,dtype: f"log2({x})", Ops.SIN: lambda x,dtype: f"sin({x})",
    Ops.AND: lambda a,b,dtype: f"({a}&{b})", Ops.XOR: lambda a,b,dtype: f"({a}^{b})", Ops.OR: lambda a,b,dtype: f"({a}|{b})",
    Ops.ADD: lambda a,b,dtype: f"({a}+{b})", Ops.SUB: lambda a,b,dtype: f"({a}-{b})", Ops.MUL: lambda a,b,dtype: f"({a}*{b})",
    Ops.MOD: lambda a,b,dtype: f"({a}%{b})", Ops.IDIV: lambda a,b,dtype: f"({a}/{b})", Ops.CMPNE: lambda a,b,dtype: f"({a}!={b})",
    Ops.SHR: lambda a,b,dtype: f"({a}>>{b})", Ops.SHL: lambda a,b,dtype: f"({a}<<{b})", Ops.CMPLT: lambda a,b,dtype: f"({a}<{b})",
    Ops.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})" }

  string_rewrite = base_rewrite
  extra_matcher = extra_pm

  def get_kernel_modifier(self, uops:List[UOp]) -> str: return ""
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501
    buftypes = [(name, self.render_dtype(dtype, mutable)+self.buffer_suffix if isinstance(dtype, (ImageDType, PtrDType)) else
                self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]
    prg = ''.join([f"{self.kernel_prefix}void {self.get_kernel_modifier(uops)}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_cast(self, dt:DType, val: str) -> str: return f"({self.render_dtype(dt)})({val})"
  def render_dtype(self, dt:DType, mutable=True) -> str:
    if isinstance(dt, ImageDType):
      return f"{'write_only' if mutable else 'read_only'} image2d_t"
    if isinstance(dt, PtrDType):
      return (self.smem_prefix if dt.local and self.smem_prefix_for_cast else self.buffer_prefix) + \
        self.render_dtype(dt.base) + ("*" if isinstance(dt, PtrDType) else "")
    return self.type_map.get(scalar:=dt.scalar(), scalar.name) + (str(dt.count) if (dt.count) > 1 else "")

  def __getitem__(self, key): return self.r[key]  # hacky helper
  def render(self, name:str, uops:List[UOp]) -> str:
    r: Dict[UOp, str] = {}
    self.r = r

    child_count = Counter(v for ru in uops for v in ru.src)
    bufs: Dict[UOp, Tuple[str, Tuple[DType, bool]]] = {}
    kernel = []
    depth = 1
    c: DefaultDict[str, int] = defaultdict(int)
    for u in uops:
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else u.arg[0]
        bufs[u] = (r[u], (u.dtype, False))
        continue

      # mark buffers that we store to writable
      if u.op is Ops.STORE:
        for up in u.src[0].toposort:
          if up.op is Ops.DEFINE_GLOBAL: bufs[up] = (bufs[up][0], (bufs[up][1][0], True))

      # naming
      prefix = None
      if u.op is Ops.SPECIAL:
        r[u] = u.arg[0]
      else:
        prefix = {Ops.RANGE: "ridx", Ops.WMMA: "wmma", Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const",
                  Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.GEP: "gep", Ops.VECTORIZE: "cast", Ops.NOOP: "precast",
                  Ops.INDEX: "bidx", Ops.DEFINE_ACC: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {Ops.ENDIF, Ops.ENDRANGE}: depth -= 1
      if u.op in {Ops.CONST, Ops.GEP, Ops.INDEX} or (u.op in {Ops.VECTORIZE, *GroupOp.ALU, Ops.CAST, Ops.BITCAST}
                                                        and child_count[u] == 1 and not getenv("EXPAND_SSA")):
        r[u] = l
      else:
        if u.op in {Ops.RANGE, Ops.ASSIGN, Ops.DEFINE_LOCAL} or u.dtype == dtypes.void:
          if u.op is Ops.ASSIGN: r[u] = r[u.src[0]]
        else:
          l = f"{self.render_dtype(u.dtype)} {r[u]} = {l}" + (";" if u.op is not Ops.SPECIAL else "")
        kernel.append("  "*depth + l)
        if prefix: c[prefix] += 1  # if it was used, increment
      if u.op in {Ops.IF, Ops.RANGE}: depth += 1
    del self.r

    # NOTE: this relies on bufs dict preserving order
    return self.render_kernel(name, kernel, list(bufs.values()), uops)

class ClangRenderer(CStyleLanguage):
  device = "CLANG"
  float4 = "(float4)"
  has_local = False
  global_max = None
  infinity = "__builtin_inff()"
  nan = '__builtin_nanf("")'

  # language options
  buffer_suffix = " restrict"
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.EXP2, Ops.SIN, Ops.LOG2]}),
                 Ops.SQRT: lambda x,dtype: f"__builtin_sqrt({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})"}

  if AMX:
    tensor_cores = [TensorCore(dims=(sz,sz,1), threads=[], reduce_axes=[], upcast_axes=([(1,sz)],[(0,sz)],[(1,sz),(0,sz)]), dtype_in=dt, dtype_out=dt)
      for dt, sz in [(dt, 64//dt.itemsize) for dt in [dtypes.float]]]

  def render_vector_prefix(self, dt:DType) -> str:
    return f"typedef {self.render_dtype(dt.scalar())} {self.render_dtype(dt)} __attribute__((aligned({(sz:=dt.itemsize)}),vector_size({sz})));"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = [self.render_vector_prefix(dt) for dt in uops_to_dtypes(uops) if dt.count > 1]
    # https://github.com/corsix/amx
    for name, (N, M, _), dtype_in, _, _, _, _, _ in dedup([uop.arg for uop in uops if uop.op is Ops.WMMA]):
      prefix += [
        '#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")',
        '#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")',
      ]
      prefix += [f"""{(out := self.render_dtype(dtype_in.vec(N*N)))} __{name}({self.render_dtype(dtype_in.vec(N))} data1, {self.render_dtype(dtype_in.vec(M))} data2, {out} data0){{
  AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}
  AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}"""] # noqa: E501
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)



class MetalRenderer(CStyleLanguage):
  device = "METAL"
  shared_max = 32768
  tensor_cores = [TensorCore(dims=(8,8,8),threads=[(0,2),(1,4),(0,2),(1,2)],expanded_shape=(2,2,2,2),upcast_axes=([(1,2)],[(1,2)],[(1,2)]),
    st1_pattern=(((1,1),(0,1),(1,0),(0,3)),((0,0),(0,2),(1,3),(1,2))),st2_pattern=(((0,0),(1,1),(1,2),(0,2),(1,0)),((0,1),(0,3),(1,3))),
    dtype_in=di,dtype_out=do,reduce_axes=[(0,8)]) for di,do in [(dtypes.float,dtypes.float),(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),
                                                                (dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]
  def __init__(self): self.tensor_cores = MetalRenderer.tensor_cores if hasattr(os, 'uname') and os.uname().machine == "arm64" else []

  # language options
  kernel_prefix = "kernel "
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}
  # uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  type_map = {dtypes.bfloat16: "bfloat"}

  # precise::sin
  code_for_op = {**CStyleLanguage.code_for_op, Ops.SIN: lambda x,dtype: f"precise::sin({x})"}

  # upcast to float32 all the ops that don't support bfloat16
  extra_matcher = PatternMatcher([
    # NOTE: this is copied from PTX
    (UPat((Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN), dtype=dtypes.bfloat16, name="x"),
      lambda x: (UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(dtypes.bfloat16))),
  ]) + extra_pm

  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"as_type<{ctx.render_dtype(x.dtype)}>({ctx[x.src[0]]})"),
  ]) + base_rewrite

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix, wmma_args = ["#include <metal_stdlib>","using namespace metal;"], set([uop.arg for uop in uops if uop.op is Ops.WMMA])
    for arg in wmma_args: prefix.append(
  f"""{(dtype_out:=self.render_dtype(arg[3].vec(2)))} __{arg[0]}({(dtype_in:=self.render_dtype(arg[2].vec(2)))} a, {dtype_in} b, {dtype_out} c){{
  simdgroup_{self.render_dtype(arg[2])}8x8 mat_a, mat_b; simdgroup_{self.render_dtype(arg[3])}8x8 mat_c;
  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return {dtype_out}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

_nms = "xyzwabcdefghijkl"
