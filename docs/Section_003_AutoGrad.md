# Section: 3 AutoGrad

PyTorch has a capability of automatic gradient calculation !

## Why we require AutoGrad ?

When we do backpropragation we need to calculate gradient of loss function w.r.t weight 

If we do gradient calculation with hands it will take time and it won't be dynamic as then we would have to write each derivative manually. 

To resolve this issue PyTorch has a capability to calculate derivative of function automatically which is also known as AutoGrad. 

!!! Info
    
    A simplified model of a PyTorch tensor is as an object containing the following properties:
    
    1. **data** — a self-reference.
    2. **required_grad** — whether or not this tensor is/should be connected to the computational graph.
    3. **grad** — if required_grad is true, this prop will be a sub-tensor that collects the gradients against this tensor accumulated during backwardpropagation.
    4. **grad_fn** — This is a reference to the most recent operation which generated this tensor. PyTorch performs automatic differentiation by looking through the grad_fn list.
    5. **is_leaf** — Whether or not this is a leaf node.


## Demo Notebooks - 

* Derivatives, Partial derivative, & Successive Differentiation - [nbviewer](https://nbviewer.org/github/c17hawke/Pytorch-basics/blob/main/codebase/03.01%20Derivatives%2C%20Partial%20derivative%2C%20and%20Successive%20Differentiation.ipynb)