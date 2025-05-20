
import jax
import jax.numpy as jnp
import equinox as eqx

import einshape as es

from examples.transformer import ResidualBlock, MultiHeadAttentionBlock


from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

import mpx



class VIT(eqx.Module):
    """Vision Transformer model from the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).
    """
    input_patch_length: Int  # in pixels
    # cls_token: Array
    num_channels: Int
    input_dim: Int
    input_residual_block: ResidualBlock
    multi_head_attention_layers: list[MultiHeadAttentionBlock]
    residual_layers: list[eqx.Module]
    output_residual_block: eqx.Module
    pos_embeddings: Array

    cls_token: Array | None


    def __init__(self,
                 output_dim: Int,
                 num_features: Int,
                 num_heads: Int,
                 num_features_residual: Int,
                 num_transformer_blocks: Int,
                 num_features_head: Int,
                 dropout_rate: Float,
                 key: PRNGKeyArray,
                 ):
        
        key, subkey = jax.random.split(key)
        self.input_patch_length = 16
        self.num_channels = 3
        input_dim = self.num_channels * self.input_patch_length**2
        self.input_dim = input_dim

        self.cls_token = jax.random.normal(key, (1, input_dim))

        self.input_residual_block = ResidualBlock(input_dim=input_dim,
                                                    output_dim=num_features, 
                                                    feature_dim=num_features_residual,
                                                    dropout_rate=dropout_rate,
                                                    num_layers=2,
                                                    transform_residual=True,
                                                    use_layernorm=False,
                                                    key=subkey,
                                                    use_residual=True,
                                                    activation="gelu"
                                                    )

        layers = []
        for _ in range(num_transformer_blocks):
            key, subkey = jax.random.split(key)
            layers.append(MultiHeadAttentionBlock(feature_dim=num_features,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate,
                                                key=subkey))
        self.multi_head_attention_layers = layers

        layers = []
        for _ in range(num_transformer_blocks):
            key, subkey = jax.random.split(key)
            layers.append(ResidualBlock(input_dim=num_features,
                                        output_dim=num_features,
                                        feature_dim=num_features_residual,  # num_features_attention * 2,
                                        dropout_rate=dropout_rate,
                                        num_layers=2,
                                        transform_residual=False,
                                        key=subkey,
                                        use_residual=True,
                                        use_layernorm=True,
                                        activation="gelu"))    
        self.residual_layers = layers

        key, subkey = jax.random.split(key)
        self.output_residual_block = ResidualBlock(input_dim=num_features,
                                                output_dim=output_dim,
                                                feature_dim=num_features_head,  # num_features_attention * 2,
                                                dropout_rate=dropout_rate * 0.0,
                                                num_layers=2,
                                                transform_residual=True,
                                                key=subkey,
                                                use_residual=False,
                                                use_layernorm=True,
                                                activation="tanh")

        self.pos_embeddings = jax.random.normal(key, (512, num_features))
                
    

    def __call__(self, inputs: Array, inference: bool, key: PRNGKeyArray) -> Array:

        # patch input
        inputs = es.jax_einshape("(np)(mq)c->(nm)(pqc)", inputs, p=self.input_patch_length, q=self.input_patch_length, c=self.num_channels)
        
        inputs = jnp.concatenate((inputs, self.cls_token), axis=0)
        
        # input residual block
        key, subkey = jax.random.split(key)
        tokens = jax.vmap(self.input_residual_block, (0, None, None))(inputs, 
                                                                    inference, 
                                                                    subkey)

        if self.pos_embeddings is not None:
            tokens += self.pos_embeddings[:len(tokens), :]
        
        # transformer layers
        for i, (multi_head_attention_layer, residual_layer) in enumerate(zip(self.multi_head_attention_layers, self.residual_layers)):
            key, subkey = jax.random.split(key)
            tokens = multi_head_attention_layer(tokens,
                                                inference=inference,
                                                key=subkey)
            
            key, subkey = jax.random.split(key)
            tokens = jax.vmap(residual_layer, (0, None, None))(tokens,
                                                            inference,
                                                            subkey)

        # output residual block
        key, subkey = jax.random.split(key)
        # ATTENTION: do not vmap over keys, as the partial layer dropout has to be the same over the time dimension.
        outputs  = jax.vmap(self.output_residual_block, (0, None, None))(tokens, 
                                                                        inference, 
                                                                        subkey)


        outputs = mpx.force_full_precision(jax.nn.softmax, outputs.dtype)(outputs[-1, :].flatten())

        return outputs

    def loss(self, pred, target):
        """Loss function for the vit.
        """
        target_one_hot = jax.nn.one_hot(target, num_classes=pred.shape[-1], dtype=pred.dtype)
        loss = -mpx.force_full_precision(jnp.sum, pred.dtype)(target_one_hot * mpx.force_full_precision(jnp.log)(pred + 1e-6))
        return loss
    
    def acc(self, pred, target):
        """Accuracy function for the vit.
        """
        pred = jnp.argmax(pred, axis=-1)
        acc = mpx.force_full_precision(jnp.mean, pred.dtype)(pred == target)
        return acc
