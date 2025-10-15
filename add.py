from tinygrad import Tensor

a = Tensor([1.1,2,2])
b = Tensor([2.2,1,2])

print((a+b).tolist())