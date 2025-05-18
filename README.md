# mpx: Mixed Precision Training for JAX

## Introduction

This repository offers a tool for training JAX models using mixed precision, called **mpx**. It builds upon [JMP](https://github.com/google-deepmind/jmp)—another mixed precision library for JAX—but extends its capabilities. 
I discovered that JMP does not support arbitrary PyTrees and is particularly incompatible with models developed using [Equinox](https://docs.kidger.site/equinox/). To overcome these limitations, I created mpx, which leverages Equinox's flexibility to work with any PyTree.

## Basics of Mixed Precision Training

This section summarizes the original Mixed Precision method from https://developer.nvidia.com/automatic-mixed-precision and https://arxiv.org/pdf/1710.03740.
Mixed Precision training involves performing most of the computations in the forward and backward passes of a neural network using 16-bit floating-point numbers.
This approach reduces GPU memory usage by roughly half compared to full precision training, allowing for larger batch sizes or the use of fewer TPUs/GPUs. Additionally, mixed precision can speed up training by decreasing memory access times and utilizing specialized half-precision tensor cores on modern hardware (if available).

One of the key factors when successfully applying Mixed Precision training is loss scaling. Due to the decreased resolution of float16, small gradients are cast to zero, decreasing training performance. The loss scaling scales the loss by a factor > 1, and as a result the gradients during gradient calculation. Afterwards, the gradients are cast to float32 and then divided by the factor to obtain the original gradient. A standard optimizer then uses the gradient to calculate the model update. The scaling can be chosen automatically with a simple heuristic. If the scaled gradients exceed the range of float16 (i.e., they are inf), we reduce the scaling and do not update the model. If the scaled gradients to not exceed the range of float16 for a longer time, we increase the scaling. 

Mixed Precision Training hence has the following steps:
1. Initialize the Model and Optimizer using Full Precision.
2. Get a Batch from the dataloader.
3. Cast the batch and model for half precision (e.g., float16 or bfloat16).
4. Do the forward pass in halfprecision, except critical operations.
5. Scale the loss.
6. Calculate the gradient of the scaled loss with respect to the weights.
8. Cast weights to float32 and divide by the scaling value.
9. If gradients are infinite, decrease scaling, else, increase scaling if in every n-th epoch.
10. If gradients are finit do optimizer update, continue with 2.

`mpx` provides important functions for steps 3--9. However, it does not provide a Keras/PyTorch Lightning/Kauldron-like functionality, where you just pass model, loss and optimizer and call run. This is done on purpose to not hurt the low-level approach of JAX and allow users to write their training pipeline like they prefer.

## Main Features

`mpx` provides a comprehensive set of tools for mixed precision training in JAX. 
The main goal was to keep the library as flexible and as close to `equinox` as possible.
As a result, to update a training pipeline to work with mixed precision, one just have to:
- Update the gradient calculations from `eqx.filter_grad/filter_value_and_grad` to `mpx.filter_grad/filter_value_and_grad`.
- Do the `optax` optimizer call via `mpx.optimizer_update`.
- Force critical operations like sum, mean and softmax to full precision using `mpx.force_full_precision`.
Here are the key components:

### Data Type Management
- `set_half_precision_datatype(dtype)`: Configure whether to use `float16` or `bfloat16` for half precision training
- `half_precision_datatype()`: Get the currently configured half precision data type

### Casting Functions
- `cast_to_half_precision(x: PyTree)`: Cast all JAX arrays in a PyTree to the configured half precision type
- `cast_to_full_precision(x: PyTree)`: Cast all JAX arrays in a PyTree to `float32`
- `cast_to_float16(x: PyTree)`: Cast all JAX arrays in a PyTree to `float16`
- `cast_to_bfloat16(x: PyTree)`: Cast all JAX arrays in a PyTree to `bfloat16`
- `cast_to_float32(x: PyTree)`: Cast all JAX arrays in a PyTree to `float32`

### Precision Control
- `force_full_precision`: A decorator that ensures a function performs all calculations in `float32`. This is essential for maintaining numerical stability in operations like mean, sum, and softmax. Currently, this has to be done by hand, i.e., **`mpx` does not identify critical operations and forces the to full precision, like AMP of PyTorch**. This unfortunately also means that provided neural network functions from equinox or flax that include these critical operations (e.g., equinox.nn.MultiheadAttention) must be rewritten by hand (please refer to our example). In a future release, we plan to provide a library that includes typical neural network functions, like Attention, that are ready for mixed precision. 

### Loss Scaling
- `DynamicLossScaling`: A class that manages dynamic loss scaling to prevent underflow in half precision training. It is syntactically equivalent to `jmp.DynamicLossScaling`, however it can scale arbitrary PyTrees.
  - `scale(x)`: Scale a value by the current loss scaling factor
  - `unscale(x)`: Remove the loss scaling factor from a value
  - `adjust(grads_finite)`: Update the loss scaling factor based on gradient stability
These functions are just for your information. They are internally used, however these might be interesting for non-standard implementations.
- `scaled(func, scaling)`: Decorator that applies loss scaling to a function's output
- `all_finite(tree)`: Check if all values in a PyTree are finite (not NaN or Inf)

### Gradient Computation
`mpx` provides function decorators for gradient calculations that summarize steps 3--9 in one function call. They have the same meaning and syntax as the corresponding decorators of `equinox`. This means, for an existing training pipeline, one can replace the calls of `equinox.filter_grad/filter_value_and_grad` with `mpx.filter_grad/filter_value_and_grad`
- `filter_grad(func, scaling: loss_scaling.DynamicLossScaling, has_aux=False, use_mixed_precision=True)`: Transformation that computes the gradient of func with respect to its first argument using mixed precision with scaling, similar to `equinox.filter_grad`. The transformed function then works as follows:
  1. If `use_mixed_precision` is True:
     - Casts all input arguments to half precision (float16/bfloat16)
     - Scales the function's output by `scaling`
  2. Computes gradients using `equinox.filter_grad`
  3. If `use_mixed_precision` is True:
     - Casts gradients back to full precision (float32)
     - Checks if gradients are finite
     - Updates `scaling` based on whether the gradients are inf or not.
     - Unscales the gradients by dividing with `scaling`
  4. Returns a tuple containing:
     - The updated `scaling` object
     - A boolean indicating if gradients are finite (needed for optimized step see below)
     - The computed gradients
     - Auxiliary values (if `has_aux=True`)

- `filter_value_and_grad(func, scaling)`: Decorator that works like `filter_grad`, except that it also returns the value.

The gradient transformations might return gradients that are infinite. In this case, the pipeline needs to skip the model update. For this, `mpx` provides the following function:
- `optimizer_update(model: PyTree, optimizer: optax.GradientTransformation, optimizer_state: PyTree, grads: PyTree, grads_finite: Bool)`: Apply optimizer updates only when gradients are finite. Works with arbitrary `optax` optimizers.

## Example
The following provides a small example, presenting all the important features of mpx. For details please examples/Bert.py.


## Acknowledgements
We want to thank Partick Kidger for providing equinox and google DeepMind for providing JMP, which was the base for this implementation.

The authors gratefully acknowledge the computing time provided to them at the NHR Center NHR4CES at RWTH Aachen University (project number p0021919). This is funded by the Federal Ministry of Education and Research, and the state governments participating on the basis of the resolutions of the GWK for national high performance computing at universities (www.nhr-verein.de/unsere-partner).


