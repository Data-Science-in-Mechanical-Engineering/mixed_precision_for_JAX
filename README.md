# MPfJ: Mixed Precision Training for JAX

**Work in Progress:**  
This library is still under development. If you’d like to use the code, feel free to copy it into your repository. I am currently benchmarking, debugging, and writing automated tests. Once I’m confident in its reliability, I will publish it on PyPI.

## Introduction

This repository offers a tool for training JAX models using mixed precision, called **MPfJ**. It builds upon [JMP](https://github.com/google-deepmind/jmp)—another mixed precision library for JAX—but extends its capabilities. 
I discovered that JMP does not support arbitrary PyTrees and is particularly incompatible with models developed using [Equinox](https://docs.kidger.site/equinox/). To overcome these limitations, I created MPfJ, which leverages Equinox’s flexibility to work with any PyTree.

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

MPfJ provides important functions for steps 3--9. However, it does not provide a Keras/PyTorch Lightning/Kauldron-like functionality, where you just pass model, loss and optimizer and call run. This is done on purpose to not hurt the low-level approach of JAX and allow users to write their training pipeline like they prefer.

## Main Features
MPfJ provides function `cast_to_half_precision`, `cast_to_full_precision`, which can cast arbitrary PyTrees to half/full precision. Every leaf in the PyTree that is a JAX Array (fulfills `equinox.is_array`) is cast. All other leafs are untouched.

MPfJ provides a decorator named `force_full_precision`. Applying this decorator ensures that the decorated function performs all calculations in `jnp.float32`. This is essential for maintaining the numerical stability of mixed precision training, as certain operations (e.g., mean, sum, softmax) should be executed in full precision to prevent overflow issues.

MPfJ provides a decorator `filter_grad`, a decorator working like `equinox.filter_jit` except that gradients are calculated using mixed precision (i.e., steps 3--9).

MPfJ provides a decorator `cond_update`, only doing updates, when gradients are finite (step 10.). This function works with arbitrary optax optimizers.

## Example
The following provides a small example, presenting all the important features of MPfJ. For details please consult the documentation TODO.


