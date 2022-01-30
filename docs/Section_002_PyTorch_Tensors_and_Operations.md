# Section: 2 PyTorch Tensors and Operations

## What is tensor?

- A kind of data structure => multidimensional arrays or matrices 
- With tensors you enocode all your parameters.

## Type Conversions

- Conversions from one datatype to another.
- Conversions from torch tensors to numpy arrays and vice versa.


---

## Demo Notebook - 
[nbviewer](https://nbviewer.org/github/c17hawke/Pytorch-basics/blob/main/codebase/2%20PyTorch%20Tensors%20and%20Operations.ipynb)

### Tensors

```py linenums="1"
import torch
import numpy as np
import os
```

```py linenums="1" 
device = "cuda" if torch.cuda.is_available() else "cpu"

device
```
>    'cuda'

```py linenums="1"
basic_tensor = torch.tensor([[1,2,3],[11,22,33]])

basic_tensor
```
>    tensor([[ 1,  2,  3],
            [11, 22, 33]])


```py linenums="1"
basic_tensor.dtype
```

>    torch.int64


```py linenums="1"
basic_tensor.device
```

>    device(type='cpu')

```py linenums="1"
basic_tensor.shape
```

>    torch.Size([2, 3])


```py linenums="1"
basic_tensor.requires_grad
```
>    False

```py linenums="1"
tensor = torch.tensor([[1,2,3],[11,22,33]],
                     dtype=torch.float,
                     device=device,
                     requires_grad=True)

tensor
```

>    tensor([[ 1.,  2.,  3.],
            [11., 22., 33.]], device='cuda:0', requires_grad=True)


```py linenums="1"
tensor.dtype
```

>    torch.float32

```py linenums="1"
tensor.device
```
>    device(type='cuda', index=0)

```py linenums="1"
tensor.requires_grad
```
>    True

#### Other commonly used tensors

```py linenums="1"
x = torch.empty(size=(3,3))
x
```
>    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])

```py linenums="1"
x = torch.zeros(size=(3,3))
x
```
>    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])

```py linenums="1"
x = torch.ones(size=(3,3))
x
```
>    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])

```py linenums="1"
x = torch.rand(size=(3,4))
x
```
>    tensor([[0.7332, 0.3748, 0.0849, 0.9105],
            [0.2788, 0.3333, 0.6220, 0.6664],
            [0.3703, 0.9297, 0.6921, 0.1396]])

```py linenums="1"
x = torch.eye(3)
x
```

>    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

```py linenums="1"
x = torch.arange(start=0, end=7, step=2)
x
```
>    tensor([0, 2, 4, 6])

```py linenums="1"
x = torch.linspace(start=0, end=7, steps=10)
x
```

>    tensor([0.0000, 0.7778, 1.5556, 2.3333, 3.1111, 3.8889, 4.6667, 5.4444, 6.2222,
            7.0000])

```py linenums="1"
x = torch.rand(size=(3,4)).normal_(mean=0, std=1)
x
```

>    tensor([[ 1.0478, -1.0514,  0.5596, -1.2438],
            [ 0.5222,  2.4026,  0.6896,  1.0098],
            [-1.0985,  0.5391,  1.9458, -1.8787]])

```py linenums="1"
x = torch.rand(size=(3,4)).uniform_(3, 6)
x
```

>    tensor([[4.5163, 5.3036, 4.8373, 5.9569],
            [4.8600, 5.1942, 5.3013, 5.2837],
            [5.7229, 5.4198, 5.7625, 4.3776]])

```py linenums="1"
x = torch.diag(torch.ones(10))
x
```

>    tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

```py linenums="1"
x.shape
```

>    torch.Size([10, 10])

```py linenums="1"
x = torch.diag(5*torch.ones(10))
x
```

>    tensor([[5., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 5., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 5., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 5., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 5., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 5., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 5., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 5., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 5., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 5.]])

### Type Conversions

```py linenums="1"
x = torch.arange(4)
x
```

>    tensor([0, 1, 2, 3])




```py linenums="1"
x.bool()
```




>    tensor([False,  True,  True,  True])




```py linenums="1"
x.int()
```




>    tensor([0, 1, 2, 3], dtype=torch.int32)




```py linenums="1"
x.short() # int16
```




>    tensor([0, 1, 2, 3], dtype=torch.int16)




```py linenums="1"
x.long() # int64
```




>    tensor([0, 1, 2, 3])




```py linenums="1"
x.half() # float16
```




>   tensor([0., 1., 2., 3.], dtype=torch.float16)




```py linenums="1"
x.float() # float32
```




>    tensor([0., 1., 2., 3.])




```py linenums="1"
x.double() # float64
```




>    tensor([0., 1., 2., 3.], dtype=torch.float64)




```py linenums="1"
np_array = np.array([[1,2,3], [1,2,3]])

np_array
```




>    array([[1, 2, 3],
           [1, 2, 3]])




```py linenums="1"
tensor = torch.from_numpy(np_array)
tensor
```




>    tensor([[1, 2, 3],
            [1, 2, 3]], dtype=torch.int32)




```py linenums="1"
tensor.numpy()
```




>    array([[1, 2, 3],
           [1, 2, 3]])

