# mpx: Mixed Precision Training for JAX



## Installation
Before installing the library, please make sure that you installed JAX for your given hardware.
```console
pip install mixed-precision-for-JAX
```

## Documentation
For basic usage, this README should give you everything you need to know. 
For deeper insights, you can read the [documentation](https://data-science-in-mechanical-engineering.github.io/mixed_precision_for_JAX/) (https://data-science-in-mechanical-engineering.github.io/mixed_precision_for_JAX/) and our [paper](https://www.arxiv.org/pdf/2507.03312) (https://www.arxiv.org/pdf/2507.03312).

## Introduction

This repository offers a tool for training JAX models using mixed precision, called **mpx**. It builds upon [JMP](https://github.com/google-deepmind/jmp)—another mixed precision library for JAX—but extends its capabilities. 
JMP does not support arbitrary PyTrees and is particularly incompatible with models developed using [Equinox](https://docs.kidger.site/equinox/). mpx overcomes these limitations, by leveraging Equinox's flexibility to work with any PyTree.

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
- `force_full_precision`: A decorator that ensures a function performs all calculations in `float32`. This is essential for maintaining numerical stability in some operations. Some critical operations in JAX, like `jax.numpy.sum/mean`, internally convert half precision to full precision. The same is true for common equinox layers like `equinox.nn.MultiheadAttention` that also force critical parts to full precision. However, this function might be useful for other implementations that do not do this.

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
The following provides a small example, training a vision transformer on Cifar100 presenting all the important features of `mpx`. For details, please visit examples/train_vit.py.
This example will not go into the details for the neural network part, but just the `mpx` relevant parts.

The example was tested on an RTX4070, the training crashes with a batch size of 256 without mixed precision. With mixed precision, the training runs, demonstrating that mixed precision training via `mpx` effectively reduces the memory used on the GPU. The training speed itself does not change dramatically as the RTX4070 does not have a higher throughput for half precision operations.

### Installation and Execution of the Example
First install JAX for your hardware.
Then, install all dependencies via
```bash
pip install -r examples/requirements.txt
```
Then you can run the example via. ATTENTION: The script downloads Cifar100.
```bash
python -m examples.train_vit
```

### Explanation
The loss scaling has to be initialized during the instantiation of the datasets, models etc. Typically, the initial value is set to the maximum value of `float16`.

```python

loss_scaling = mpx.DynamicLossScaling(loss_scaling=mpx.FLOAT16_MAX, 
                                      min_loss_scaling=jnp.ones((), dtype=jnp.float32), 
                                      period=2000)
```
The loss_scaling object then must be passed to the training pipeline.

The most important part is the training step. `mpx` makes transforming your training step into mixed precision very easy. As you can see, the only change you have to do is to replace a call to `eqx.filter_value_and_grad` with `mpx.filter_value_and_grad` and afterwards call the optimizer via `mpx.optimizer_update`. Also, do not forget to return `loss_scaling` in your step function, because `loss_scaling` is updated.

```python
@eqx.filter_jit
def make_step(model: eqx.Module, 
            optimizer: any, 
            optimizer_state: PyTree, 
            batch: dict,
            batch_sharding: jax.sharding.NamedSharding,
            replicated_sharding: jax.sharding.NamedSharding,
            loss_scaling: mpx.DynamicLossScaling,
            train_mixed_precicion: bool,
            weight_regularization: Float,
            key: PRNGKeyArray
              ) -> tuple[eqx.Module, PyTree, Float, PRNGKeyArray]:
    batch = eqx.filter_shard(batch, batch_sharding)
    model = eqx.filter_shard(model, replicated_sharding)
    optimizer_state = eqx.filter_shard(optimizer_state, replicated_sharding)
    

    if train_mixed_precicion:
        # this is the critical part
        (loss_value, _), loss_scaling, grads_finite, grads = mpx.filter_value_and_grad(batched_loss_acc_wrapper, scaling=loss_scaling, has_aux=True)(
            model, batch, batch_sharding, replicated_sharding, key, weight_regularization)
        model, optimizer_state = mpx.optimizer_update(model, optimizer, optimizer_state, grads,grads_finite)
    else:
        (loss_value, _), grads = eqx.filter_value_and_grad(batched_loss_acc_wrapper, has_aux=True)(
            model, batch, batch_sharding, replicated_sharding, key)
        # optimizer step
        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

    model = eqx.filter_shard(model, replicated_sharding)
    optimizer_state = eqx.filter_shard(optimizer_state, replicated_sharding)
    loss_scaling = eqx.filter_shard(loss_scaling, replicated_sharding)
    
    # return loss_scaling as it is changed
    return model, optimizer_state, loss_scaling, loss_value
```

Through the transformation via `mpx.filter_value_and_grad`, we can write our loss function as we normally do when using JAX/Equinox.

## Citation

To cite this repository, please cite our [paper](https://www.arxiv.org/pdf/2507.03312):

```
@ARTICLE{2025arXiv250703312G,
  author = {{Gr{\"a}fe}, Alexander and {Trimpe}, Sebastian},
  title = "{MPX: Mixed Precision Training for JAX}",
  journal = {arXiv e-prints},
  year = 2025,
  doi = {10.48550/arXiv.2507.03312},
}


``` 

## Acknowledgements
We want to thank Partick Kidger for providing equinox and google DeepMind for providing JMP, which was the base for this implementation.

The authors gratefully acknowledge the computing time provided to them at the NHR Center NHR4CES at RWTH Aachen University (project number p0021919). This is funded by the Federal Ministry of Education and Research, and the state governments participating on the basis of the resolutions of the GWK for national high performance computing at universities (www.nhr-verein.de/unsere-partner).


