"""
Implements the basics of transformers.
All modules are for non-batched inputs. To create calls for batched data use vmap.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import einshape as es

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

import mpx



def init_weights(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    lim = 1 / jnp.sqrt(shape[0])
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


class DenseLayer(eqx.Module):
    weights: Array
    bias: Array

    def __init__(self, 
                 input_dim: Int, 
                 output_dim: Int, 
                 key: PRNGKeyArray):
        key, subkey = jax.random.split(key)
        self.weights = init_weights(subkey, (input_dim, output_dim))
        self.bias = jnp.zeros((output_dim,))

    def __call__(self, inputs: Array) -> Array:
        return inputs @ self.weights + self.bias


class ResidualBlock(eqx.Module):
    """A residual block module that applies a series of dense layers, dropout, and layer normalization 
    with an optional transformed residual connection.

    i.e. y = LayerNorm(MLP(x) + Residual(x)), where Resiudual(x) = x if transform_residual is False and Residual(x) = DenseLayer(x) if transform_residual is True
    """

    layers: list[DenseLayer]
    residual_layer: DenseLayer | None
    dropout: eqx.nn.Dropout

    layer_norm: eqx.nn.LayerNorm | None

    activation_fn: callable

    def __init__(self, 
                input_dim: Int,
                output_dim: Int,
                feature_dim: Int,
                dropout_rate: Float,
                num_layers: Int,
                transform_residual: bool,
                key: PRNGKeyArray,
                use_residual: bool,
                use_layernorm: bool,
                activation: str):

        # init layers
        layers = []
        pruning_masks = []
        key, subkey = jax.random.split(key)
        layers.append(DenseLayer(input_dim, feature_dim, subkey))

        if num_layers >= 2:
            for _ in range(num_layers - 2):
                key, subkey = jax.random.split(key)
                layers.append(DenseLayer(feature_dim, 
                                    feature_dim,
                                    subkey))
            
            key, subkey = jax.random.split(key)
            layers.append(DenseLayer(feature_dim, output_dim, subkey))
        
        self.layers = layers

        assert transform_residual or input_dim == output_dim, "input_dim must be equal to output_dim if transform_residual is False"
        if transform_residual:
            self.residual_layer = DenseLayer(input_dim, output_dim, key)
        else:
            self.residual_layer = None

        # init layernorm and dropout
        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.layer_norm = eqx.nn.LayerNorm(input_dim)

        if activation == "relu":
            self.activation_fn = jax.nn.relu
        elif activation == "tanh":
            self.activation_fn = jax.nn.tanh
        elif activation == "gelu":
            self.activation_fn = jax.nn.gelu

    def __call__(self, inputs: Array, inference: bool, key: PRNGKeyArray) -> Array:
        # first layer 
        if self.layer_norm is not None:
            inputs_after_layernorm = mpx.force_full_precision(self.layer_norm, inputs.dtype)(inputs)

            outputs = self.layers[0](inputs_after_layernorm)
        else:
            outputs = self.layers[0](inputs)
        
        if len(self.layers) >= 2:
            outputs = self.activation_fn(outputs)

            for i, layer in enumerate(self.layers[1:-1]):
                key, subkey = jax.random.split(key)
                outputs = self.activation_fn(layer(outputs))

            outputs = self.layers[-1](outputs)

            outputs = self.dropout(outputs, inference=inference, key=key)

            if self.residual_layer is not None:
                residual = self.residual_layer(inputs)
                outputs = outputs + residual
            else:
                outputs = outputs + inputs 
            return outputs
        else:
            return outputs
    

class MultiHeadAttentionBlock(eqx.Module):
    dense_qs: DenseLayer   # (num_heads * feature_dim, inner_dim), i.e., w_q of head 2 is dense_qs.weight[feature_dim:2*feature_dim, :]
    dense_ks: DenseLayer
    dense_vs: DenseLayer

    dense_o: DenseLayer

    num_heads: int

    dropout: eqx.nn.Dropout
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, feature_dim: int, 
                 num_heads: int, 
                 dropout_rate: float, 
                 key: PRNGKeyArray):
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.num_heads = num_heads

        key, subkey = jax.random.split(key)
        self.dense_qs = DenseLayer(feature_dim, feature_dim, subkey)

        key, subkey = jax.random.split(key)
        self.dense_ks = DenseLayer(feature_dim, feature_dim, subkey)

        key, subkey = jax.random.split(key)
        self.dense_vs = DenseLayer(feature_dim, feature_dim, subkey)

        key, subkey = jax.random.split(key)
        self.dense_o = DenseLayer(feature_dim, feature_dim, subkey)

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.layer_norm = eqx.nn.LayerNorm(feature_dim)

    @staticmethod
    def attention(q: Array,
                  k: Array,
                  v: Array,
                  dropout: eqx.nn.Dropout,
                  key: PRNGKeyArray, 
                  inference: bool) -> Array:
        attention_scores = q @ k.T
        attention_scores /= jnp.sqrt(q.shape[-1])
        attention_scores = mpx.force_full_precision(jax.nn.softmax, attention_scores.dtype)(attention_scores, axis=-1)
        attention_scores = dropout(attention_scores, inference=inference, key=key)
        return attention_scores @ v

    def __call__(self, inputs: Array, inference: bool, key: PRNGKeyArray) -> Array:
        inputs_after_layernorm = jax.vmap(self.layer_norm)(inputs)
        qs = jax.vmap(self.dense_qs)(inputs_after_layernorm)
        ks = jax.vmap(self.dense_ks)(inputs_after_layernorm)
        vs = jax.vmap(self.dense_vs)(inputs_after_layernorm)

        # reshape such that the first dimension is the head, the second the time and the third the features
        # The nth num_heads / num_devices heads belong to the nth device
        # As we combine the heads in the last dimension, for the input, the first num_heads / num_devices * num_features
        # belong to the first device, and so on. This is important as we apply the same pruning mask as for dense layers.
        # When reshaping, we need to take these chunkks (i.e., result[:, 0: num_heads / num_devices * num_features]) and put them into
        # the coresponding heads (i.e. to result[0:num_heads / num_devices, 0: num_features]).
        # This is exactly what the einshape call does.
        # result = es.jax_einshape("n(hf)->hnf", result, h=self.num_heads)
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
