from tinygrad import Tensor,dtypes

a = Tensor([1,2,2],dtype=dtypes.float)
b = Tensor([2,1,66],dtype=dtypes.float)

print((a+b).tolist())