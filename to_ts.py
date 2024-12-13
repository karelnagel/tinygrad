import tinygrad as tiny
import itertools
from tinygrad.ops import Ops, UPat


def trycatch(fn):
    try:
        return fn()
    except Exception as e:
        return str(e)


def to_ts(o):
    if isinstance(o, Ops):
        return f"{o.value}"
    if isinstance(o, UPat):
        src = (
            f"[{to_ts(o._in_src)}]" if isinstance(o._in_src, list) else to_ts(o._in_src)
        )
        return f"new UPat({{ op:{to_ts(o.op)}, dtype:{to_ts(o.dtype)}, src:{src}, arg:{to_ts(o.arg)}, name:{to_ts(o.name)}, allow_any_len:{to_ts(o.allowed_len == -1)}, location:{to_ts(o.location)}, custom_early_reject:{to_ts(o.custom_early_reject)} }})"
    if isinstance(o, tiny.ops.UOp):
        return f"new UOp({{ op:{to_ts(o.op)}, dtype:{to_ts(o.dtype)}, src:{to_ts(o.src)}, arg:{to_ts(o.arg)} }})"
    if isinstance(o, tiny.dtype.ImageDType):
        return f"new ImageDType({{ priority:{to_ts(o.priority)}, itemsize:{to_ts(o.itemsize)}, name:{to_ts(o.name)}, fmt:{to_ts(o.fmt)}, count:{to_ts(o.count)}, _scalar:{to_ts(o._scalar)}, _base:{to_ts(o.base)}, local:{to_ts(o.local)}, v:{to_ts(o.v)}, shape:{to_ts(o.shape)} }})"
    if isinstance(o, tiny.dtype.PtrDType):
        return f"new PtrDType({{ priority:{to_ts(o.priority)}, itemsize:{o.itemsize}, name:{to_ts(o.name)}, fmt:{to_ts(o.fmt)}, count:{to_ts(o.count)}, _scalar:{to_ts(o._scalar)}, _base:{to_ts(o.base)}, local:{to_ts(o.local)}, v:{to_ts(o.v)} }})"
    if isinstance(o, tiny.dtype.DType):
        return f"new DType({{ priority:{to_ts(o.priority)}, itemsize:{to_ts(o.itemsize)}, name:{to_ts(o.name)}, fmt:{to_ts(o.fmt)}, count:{to_ts(o.count)}, _scalar:{to_ts(o._scalar)} }})"
    if isinstance(o, tiny.shape.view.View):
        return f"new View({{ shape:{to_ts(o.shape)}, strides:{to_ts(o.strides)}, offset:{to_ts(o.offset)}, mask:{to_ts(o.mask)}, contiguous:{to_ts(o.contiguous)} }})"
    if isinstance(o, tiny.shape.shapetracker.ShapeTracker):
        return f"new ShapeTracker({to_ts(o.views)})"

    if isinstance(o, itertools.repeat):
        return to_ts(next(o))
    if callable(o):
        return "undefined"
    if isinstance(o, set):
        return f"[{', '.join(map(to_ts, o))}]"
    if isinstance(o, (list, tuple)):
        return f"[{', '.join(map(to_ts, o))}]"
    if o == float('inf'):
        return "Infinity"
    if o == float('-inf'):
        return "-Infinity"
    if isinstance(o, float) and str(o) == 'nan':
        return "NaN"
    if isinstance(o, (bool, int, float)):
        return str(o).lower()
    if o is None:
        return "undefined"
    if isinstance(o, str):
        return f'`{o}`'
    if isinstance(o, dict):
        return f"new Map([{', '.join(f'[{to_ts(k)}, {to_ts(v)}]' for k,v in o.items())}])"
    return str(o)
