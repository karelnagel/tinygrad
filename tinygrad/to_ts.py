from enum import Enum
import itertools


def to_ts(o):
    from tinygrad.ops import KernelInfo, UOp, UPat
    from tinygrad.codegen.lowerer import IndexContext
    from tinygrad.dtype import DType, ImageDType, PtrDType
    from tinygrad.shape.shapetracker import ShapeTracker
    from tinygrad.shape.view import View
    from tinygrad.renderer.cstyle import ClangRenderer
    from tinygrad.codegen.kernel import Kernel, Opt
    from tinygrad.codegen.linearize import BasicBlock
    from tinygrad.renderer import ProgramSpec, TensorCore
    from tinygrad.helpers import Metadata
    from tinygrad.device import (
        _Device,
        # _MallocAllocator,
        Allocator,
        Buffer,
        BufferSpec,
        Compiler,
        # LRUAllocator,
    )
    from tinygrad.engine.lazy import LazyBuffer
    from tinygrad.engine.realize import CompiledRunner, ExecItem, Runner
    from tinygrad.engine.schedule import (
        ScheduleContext,
        ScheduleItem,
        ScheduleItemContext,
    )
    from tinygrad.runtime.ops_python import PythonRenderer

    if isinstance(o, Enum):
        return f"{o.__class__.__name__ }.{o.name}"

    # ************ OPS ************
    if isinstance(o, UPat):
        src = (
            f"[{to_ts(o._in_src)}]" if isinstance(o._in_src, list) else to_ts(o._in_src)
        )
        return f"new UPat({to_ts(o.op)}, {to_ts(o.dtype)}, {src}, {to_ts(o.arg)}, {to_ts(o.name)}, {to_ts(o.allowed_len == -1)}, {to_ts(o.location)}, {to_ts(o.custom_early_reject)})"
    if isinstance(o, UOp):
        return (
            f"new UOp({to_ts(o.op)}, {to_ts(o.dtype)}, {to_ts(o.src)}, {to_ts(o.arg)})"
        )
    if isinstance(o, KernelInfo):
        return f"new KernelInfo({to_ts(o.local_dims)}, {to_ts(o.upcasted)}, {to_ts(o.dont_use_locals)})"

    # ************ DTYPE ************
    if isinstance(o, ImageDType):
        return f"new ImageDType({to_ts(o.priority)}, {to_ts(o.itemsize)}, {to_ts(o.name)}, {to_ts(o.fmt)}, {to_ts(o.count)}, {to_ts(o._scalar)}, {to_ts(o.base)}, {to_ts(o.local)}, {to_ts(o.v)}, {to_ts(o.shape)})"
    if isinstance(o, PtrDType):
        return f"new PtrDType({to_ts(o.priority)}, {to_ts(o.itemsize)}, {to_ts(o.name)}, {to_ts(o.fmt)}, {to_ts(o.count)}, {to_ts(o._scalar)}, {to_ts(o.base)}, {to_ts(o.local)}, {to_ts(o.v)})"
    if isinstance(o, DType):
        return f"new DType({to_ts(o.priority)}, {to_ts(o.itemsize)}, {to_ts(o.name)}, {to_ts(o.fmt)}, {to_ts(o.count)}, {to_ts(o._scalar)})"

    # ************ VIEW ************
    if isinstance(o, View):
        return f"new View({to_ts(o.shape)}, {to_ts(o.strides)}, {to_ts(o.offset)}, {to_ts(o.mask)}, {to_ts(o.contiguous)})"
    if isinstance(o, ShapeTracker):
        return f"new ShapeTracker({to_ts(o.views)})"

    # ************ RENDERER ************
    if isinstance(o, ClangRenderer):
        return f"new ClangRenderer()"
    if isinstance(o, PythonRenderer):
        return f"new PythonRenderer()"
    if isinstance(o, TensorCore):
        return f"new TensorCore({to_ts(o.dims)}, {to_ts(o.dtype_in)}, {to_ts(o.dtype_out)}, {to_ts(o.threads)}, {to_ts(o.reduce_axes)}, {to_ts(o.upcast_axes)})"
    if isinstance(o, ProgramSpec):
        return f"new ProgramSpec({to_ts(o.name)}, {to_ts(o.src)}, {to_ts(o.device)}, {to_ts(o.uops)}, {to_ts(o.mem_estimate)}, {to_ts(o.global_size)}, {to_ts(o.local_size)}, {to_ts(o.vars)}, {to_ts(o.globals)}, {to_ts(o.outs)}, {to_ts(o._ran_post_init)})"

    # ************ CODEGEN ************
    if isinstance(o, IndexContext):
        return (
            f"new IndexContext({to_ts(o.idxs)}, {to_ts(o.ridxs)}, {to_ts(o.acc_num)})"
        )
    if isinstance(o, Kernel):
        return f"new Kernel({to_ts(o.ast)}, {to_ts(o.opts)})"
    if isinstance(o, BasicBlock):
        return f"new BasicBlock({to_ts(o.ctx)}, {to_ts(o.lst)}, {to_ts(o.end)})"
    if isinstance(o, Opt):
        return f"new Opt({to_ts(o.op)}, {to_ts(o.axis)}, {to_ts(o.amt)})"

    # ************ DEVICE ************
    if isinstance(o, _Device):
        return f"new _Device()"
    if isinstance(o, BufferSpec):
        return f"new BufferSpec({to_ts(o.image)}, {to_ts(o.uncached)}, {to_ts(o.cpu_access)}, {to_ts(o.host)}, {to_ts(o.nolru)}, {to_ts(o.external_ptr)})"
    if isinstance(o, Buffer):
        return f"new Buffer({to_ts(o.device)}, {to_ts(o.size)}, {to_ts(o.dtype)}, {to_ts(o.in_opaque)}, {to_ts(o.options)}, {to_ts(o.in_initial_value)}, {to_ts(getattr(o, '_lb_refcount', None))}, {to_ts(o._base)}, {to_ts(o.offset)}, {to_ts(o.in_preallocate)})"
    if isinstance(o, Allocator):
        return f"new Allocator()"
    # if isinstance(o, LRUAllocator):
    #     return f"new LRUAllocator()"
    # if isinstance(o, _MallocAllocator):
    #     return f"new _MallocAllocator()"
    if isinstance(o, Compiler):
        return f"new Compiler({to_ts(o.cachekey)})"

    # ************ ENGINE ************
    if isinstance(o, LazyBuffer):
        return f"new LazyBuffer({to_ts(o.device)}, {to_ts(o.st)}, {to_ts(o.dtype)}, {to_ts(getattr(o,"op",None))}, {to_ts(getattr(o,"arg",None))}, {to_ts(getattr(o,"srcs",None))}, {to_ts(o._base)}, {to_ts(o.metadata)})"

    if isinstance(o, CompiledRunner):
        return f"new CompiledRunner({to_ts(o.p)}, {to_ts(o.lib)})"
    if isinstance(o, Runner):
        return f"new Runner({to_ts(o.display_name)}, {to_ts(o.device)}, {to_ts(o.op_estimate)}, {to_ts(o.mem_estimate)}, {to_ts(o.lds_estimate)})"
    if isinstance(o, ExecItem):
        return f"new ExecItem({to_ts(o.prg)}, {to_ts(o.bufs)}, {to_ts(o.metadata)})"

    if isinstance(o, ScheduleItem):
        return f"new ScheduleItem({to_ts(o.ast)}, {to_ts(o.bufs)}, {to_ts(o.metadata)}, {to_ts(o.assign_preloads)})"
    if isinstance(o, ScheduleContext):
        return f"new ScheduleContext({to_ts(o.lazybufs)}, {to_ts(o.var_vals)}, {to_ts(o.assigns)}, {to_ts(o.realizes)}, {to_ts(o.allbufs)}, {to_ts(o.ops_metadata)}, {to_ts(o.children)})"
    if isinstance(o, ScheduleItemContext):
        return f"new ScheduleItemContext({to_ts(o.lazybufs)}, {to_ts(o.ops_metadata)}, {to_ts(o.assigns)}, {to_ts(o.var_vals)}, {to_ts(o.sinked)}, {to_ts(o.sts)}, {to_ts(o.bufs)}, {to_ts(o.metadata)}, {to_ts(o.assign_adj)})"

    if isinstance(o, Metadata):
        return f"new Metadata({to_ts(o.name)}, {to_ts(o.caller)}, {to_ts(o.backward)})"

    # if hasattr(o, "__dataclass_fields__"):
    #     fields = {k: getattr(o, k) for k in o.__dataclass_fields__}
    #     return f"{{ {', '.join(f'{k}:{to_ts(v)}' for k,v in fields.items())} }}"
    if isinstance(o, bytes):
        return f"new Uint8Array([{','.join(str(x) for x in o)}])"
    if isinstance(o, memoryview):
        return f"new DataView(new Uint8Array([{','.join(str(x) for x in o)}]).buffer)"
    if isinstance(o, itertools.repeat):
        return to_ts(next(o))
    if callable(o):
        return "undefined"
    if isinstance(o, set):
        return f"new Set([{', '.join(map(to_ts, o))}])"
    if isinstance(o, frozenset):
        return f"new Set([{', '.join(map(to_ts, o))}])"
    if isinstance(o, (list, tuple)):
        return f"[{', '.join(map(to_ts, o))}]"
    if o == float("inf"):
        return "Infinity"
    if o == float("-inf"):
        return "-Infinity"
    if isinstance(o, float) and str(o) == "nan":
        return "NaN"
    if isinstance(o, (bool, int, float)):
        return str(o).lower()
    if o is None:
        return "undefined"
    if isinstance(o, str):
        return f"`{o.replace('\n', '\\n')}`"
    if isinstance(o, dict):
        return (
            f"new Map([{', '.join(f'[{to_ts(k)}, {to_ts(v)}]' for k,v in o.items())}])"
        )
    raise ValueError(f"No ts coverter for {o}")


global_inputs = {}


def save_input(fn_name, input):
    ts = to_ts(input)
    fn_inputs: set = global_inputs.setdefault(fn_name, set())
    if ts not in fn_inputs:
        if len(fn_inputs) >= 6:
            # Remove longest string if new one is shorter
            longest = max(fn_inputs, key=len)
            if len(ts) < len(longest):
                fn_inputs.remove(longest)
                fn_inputs.add(ts)
        else:
            fn_inputs.add(ts)
        with open(f"input_{fn_name}.txt", "w") as f:
            f.write(",\n".join(sorted(fn_inputs, key=len)) + "\n")