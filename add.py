from tinygrad import Tensor,dtypes

a = Tensor([1,2,2],dtype=dtypes.int)
b = Tensor([2,1,2],dtype=dtypes.int)

print((a*b).tolist())