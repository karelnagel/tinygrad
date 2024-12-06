from __future__ import annotations
import math
from typing import Optional, Union, Tuple, List
from tinygrad.tensor import Tensor, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import prod, make_tuple, flatten
from tinygrad.nn import optim, state, datasets  # noqa: F401

class BatchNorm:
  """
  Applies Batch Normalization over a 2D or 3D input.

  - Described: https://paperswithcode.com/method/batch-normalization
  - Paper: https://arxiv.org/abs/1502.03167v3

  See: `Tensor.batchnorm`

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  np.set_printoptions(precision=4)
  ```

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.BatchNorm(3)
  t = Tensor.rand(2, 3, 4, 4)
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Optional[Tensor] = Tensor.ones(sz) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(sz) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype='long' if is_dtype_supported(dtypes.long) else 'int', requires_grad=False)
    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)

  def calc_stats(self, x:Tensor) -> Tuple[Tensor, Tensor]:
    shape_mask: List[int] = [1, -1, *([1]*(x.ndim-2))]
    if self.track_running_stats and not Tensor.training: return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
    # This requires two full memory accesses to x
    # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
    # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x != 1)))
    y = (x - batch_mean.detach().reshape(shape=shape_mask))  # d(var)/d(mean) = 0
    batch_var = (y*y).mean(axis=reduce_axes)
    return batch_mean, batch_var

  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = self.calc_stats(x)
    # NOTE: wow, this is done all throughout training in most PyTorch models
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * x.numel()/(x.numel()-x.shape[1]) * batch_var.detach())
      self.num_batches_tracked += 1
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt())
BatchNorm2d = BatchNorm3d = BatchNorm

def Conv1d(in_channels:int, out_channels:int, kernel_size:int, stride=1, padding:Union[int, str]=0, dilation=1, groups=1, bias=True) -> Conv2d:
  """
  Applies a 1D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv1d(1, 1, 3)
  t = Tensor.rand(1, 1, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)

class Conv2d:
  """
  Applies a 2D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv2d(1, 1, 3)
  t = Tensor.rand(1, 1, 4, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_channels:int, out_channels:int, kernel_size:Union[int, Tuple[int, ...]], stride=1, padding:Union[int, Tuple[int, ...], str]=0,
               dilation=1, groups=1, bias=True):
    self.kernel_size = make_tuple(kernel_size, 2)
    # TODO: not needed for mnist
    # if isinstance(padding, str):
    #   if padding.lower() != 'same': raise ValueError(f"Invalid padding string {padding!r}, only 'same' is supported")
    #   if stride != 1: raise ValueError("padding='same' is not supported for strided convolutions")
    #   pad = [(d*(k-1)//2, d*(k-1) - d*(k-1)//2) for d,k in zip(make_tuple(dilation, len(self.kernel_size)), self.kernel_size[::-1])]
    #   padding = tuple(flatten(pad))
    self.stride, self.dilation, self.groups, self.padding = stride, dilation, groups, padding
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
    self.bias: Optional[Tensor] = Tensor.uniform(out_channels, low=-scale, high=scale) if bias else None

  def __call__(self, x:Tensor) -> Tensor: return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)



class Linear:
  """
  Applies a linear transformation to the incoming data.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Linear

  ```python exec="true" source="above" session="tensor" result="python"
  lin = nn.Linear(3, 4)
  t = Tensor.rand(2, 3)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = lin(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_features:int, out_features:int, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor) -> Tensor: return x.linear(self.weight.transpose(), self.bias)


class LayerNorm:
  """
  Applies Layer Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/layer-normalization
  - Paper: https://arxiv.org/abs/1607.06450v1

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm(3)
  t = Tensor.rand(2, 5, 3) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps=1e-5, elementwise_affine=True):
    self.normalized_shape: Tuple[int, ...] = make_tuple(normalized_shape, 1)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight: Optional[Tensor] = Tensor.ones(*self.normalized_shape) if elementwise_affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(*self.normalized_shape) if elementwise_affine else None

  def __call__(self, x:Tensor) -> Tensor:
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    x = x.layernorm(eps=self.eps, axis=self.axis)
    if not self.elementwise_affine: return x
    return x * self.weight + self.bias

class LayerNorm2d(LayerNorm):
  """
  Applies Layer Normalization over a mini-batch of 2D inputs.

  See: `LayerNorm`

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm2d(3)
  t = Tensor.rand(2, 3, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __call__(self, x: Tensor) -> Tensor: return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class RMSNorm:
  """
  Applies Root Mean Square Normalization to input.

  - Described: https://paperswithcode.com/method/rmsnorm
  - Paper: https://arxiv.org/abs/1910.07467

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.RMSNorm(4)
  t = Tensor.arange(12, dtype=dtypes.float).reshape(3, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  print(norm(t).numpy())
  ```
  """
  def __init__(self, dim:int, eps=1e-6): self.eps, self.weight = eps, Tensor.ones(dim)

  def _norm(self, x:Tensor) -> Tensor: return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()

  def __call__(self, x:Tensor) -> Tensor: return self._norm(x.float()).cast(x.dtype) * self.weight
