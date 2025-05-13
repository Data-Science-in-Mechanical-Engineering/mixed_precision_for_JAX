import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int
import einshape as es

import mpfj

class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=output_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, " hidden_size"],
        enable_dropout: bool = True,
        key: jax.random.PRNGKey | None = None,
    ) -> Float[Array, " hidden_size"]:
        # Feed-forward.
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)

        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual and layer norm.
        output += inputs
        # layernorm contains mean and std, so we need to force full precision
        output = mpfj.force_full_precision(self.layernorm, return_dtype=output.dtype)(output)

        return output
    

class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    W_q: Array
    b_q: Array
    W_k: Array
    b_k: Array
    W_v: Array
    b_v: Array
    W_o: Array
    b_o: Array
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, " seq_len"] | None,
        enable_dropout: bool = False,
        key: "jax.random.PRNGKey" = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        Q = inputs @ self.W_q + self.b_q
        K = inputs @ self.W_k + self.b_k
        V = inputs @ self.W_v + self.b_v

        Q_head = es.jax_einshape("n(hd)->hnd", Q, h=self.num_heads)
        K_head = es.jax_einshape("n(hd)->hnd", K, h=self.num_heads)
        V_head = es.jax_einshape("n(hd)->hnd", V, h=self.num_heads)

        def _single_head_attention(q, k, v):
            qk = q @ k.T
            qk = qk / jnp.sqrt(Q.shape[-1])
            qk = qk * mask
            # softmax contains exp, mean and division, so we need to force full precision
            qk = mpfj.force_full_precision(jax.nn.softmax, return_dtype=qk.dtype)(qk)
            o = qk @ v
            return o
        
        O = jax.vmap(_single_head_attention, in_axes=(0, 0, 0))(Q_head, K_head, V_head)

        result = es.jax_einshape("hnd->n(hd)", O, h=self.num_heads) @ self.W_o + self.b_o

        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        # layernorm contains mean and std, so we need to force full precision
        result = mpfj.force_full_precision(jax.vmap(self.layernorm), return_dtype=result.dtype)(result)
        return result

    def make_self_attention_mask(
        self, mask: Int[Array, " seq_len"]
    ) -> Float[Array, "num_heads seq_len seq_len"]:
        """Create self-attention mask from sequence-level mask."""
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2)
        )
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)
    

class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, " seq_len"] | None = None,
        *,
        enable_dropout: bool = False,
        key: jax.random.PRNGKey | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(
            inputs, mask, enable_dropout=enable_dropout, key=attn_key
        )
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )
        return output
