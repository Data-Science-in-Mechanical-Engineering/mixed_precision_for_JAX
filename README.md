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
Only critical operations, like mean need to be forced to full precision.
```python
@eqx.filter_jit
def batched_loss_acc_wrapper(model, batch, batch_sharding, replicated_sharding, key, weight_regularization=0.0):
    batch = eqx.filter_shard(batch, batch_sharding)
    model = eqx.filter_shard(model, replicated_sharding)

    pred = predict_batch(model, batch, False, key)

    target = batch["target"]    
    losses = jax.vmap(model.loss)(pred, target)
    acc = jax.vmap(model.acc)(pred, target)

    # especially for high batch sizes the mean calculation can overflow, hence force it to full precision.
    loss = mpx.force_full_precision(jnp.mean, losses.dtype)(losses)
    acc = mpx.force_full_precision(jnp.mean, losses.dtype)(acc)

    # weight regularization can help in mixed precision training.
    # It keeps the weights small and prevents overflow during matrix multiplication.
    params, _ = eqx.partition(model, eqx.is_array)
    params = jax.tree_util.tree_leaves(params)
    params = jax.tree_util.tree_map(lambda x: x.flatten(), params)
    params = jnp.concatenate(params).flatten()

    loss = loss + weight_regularization * mpx.force_full_precision(jnp.mean, jnp.float32)(jnp.abs(params))

    return loss, acc
```

The same holds for the neural network. Here, only critical operations like layernorm and softmax must be forced to full precision.
This means, as long as your layer does not contain such operations, you can rely on standard `equinox.nn` implementations. For other layers, the only solution so far is to reimplement them and force critical operations to full precision.

```python
class MultiHeadAttentionBlock(eqx.Module):
    dense_qs: DenseLayer
    dense_ks: DenseLayer
    dense_vs: DenseLayer

    dense_o: DenseLayer

    num_heads: int

    dropout: eqx.nn.Dropout
    layer_norm: eqx.nn.LayerNorm

    ...

    @staticmethod
    def attention(q: Array,
                  k: Array,
                  v: Array,
                  dropout: eqx.nn.Dropout,
                  key: PRNGKeyArray, 
                  inference: bool) -> Array:
        attention_scores = q @ k.T
        attention_scores /= jnp.sqrt(q.shape[-1])

        # softmax is critical
        attention_scores = mpx.force_full_precision(jax.nn.softmax, attention_scores.dtype)(attention_scores, axis=-1)

        attention_scores = dropout(attention_scores, inference=inference, key=key)
        return attention_scores @ v

    def __call__(self, inputs: Array, inference: bool, key: PRNGKeyArray) -> Array:
        # also force layernorm to full precision.
        inputs_after_layernorm = jax.vmap(mpx.force_full_precision(self.layer_norm, inputs.dtype))(inputs)
        qs = jax.vmap(self.dense_qs)(inputs_after_layernorm)
        ks = jax.vmap(self.dense_ks)(inputs_after_layernorm)
        vs = jax.vmap(self.dense_vs)(inputs_after_layernorm)

        qs = es.jax_einshape("n(hf)->hnf", qs, h=self.num_heads)
        ks = es.jax_einshape("n(hf)->hnf", ks, h=self.num_heads)
        vs = es.jax_einshape("n(hf)->hnf", vs, h=self.num_heads)

        keys = jax.random.split(key, self.num_heads)

        outputs = jax.vmap(self.attention, in_axes=(0, 0, 0, None, 0, None))(
            qs, 
            ks,
            vs,
            self.dropout,
            keys,
            inference)

        # reshape outputs (concatenate heads)
        outputs = es.jax_einshape("hnf->n(hf)", outputs)

        key, key2 = jax.random.split(key)
        outputs = jax.vmap(self.dense_o)(outputs)
        outputs = self.dropout(outputs, inference=inference, key=key2)
        outputs += inputs

        return outputs
```

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


