"""
This is a port of the Bert example from Equinox (https://docs.kidger.site/equinox/examples/bert/).
"""

import functools
from collections.abc import Mapping

import einops  # https://github.com/arogozhnikov/einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax  # https://github.com/deepmind/optax
from datasets import load_dataset  # https://github.com/huggingface/datasets
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping
from tqdm import notebook as tqdm  # https://github.com/tqdm/tqdm
from transformers import AutoTokenizer  # https://github.com/huggingface/transformers

import einshape as es

from examples.transformer import TransformerLayer

import mpx

class EmbedderBlock(eqx.Module):
    """BERT embedder."""

    token_embedder: eqx.nn.Embedding
    segment_embedder: eqx.nn.Embedding
    position_embedder: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        type_vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        token_key, segment_key, position_key = jax.random.split(key, 3)

        self.token_embedder = eqx.nn.Embedding(
            num_embeddings=vocab_size, embedding_size=embedding_size, key=token_key
        )
        self.segment_embedder = eqx.nn.Embedding(
            num_embeddings=type_vocab_size,
            embedding_size=embedding_size,
            key=segment_key,
        )
        self.position_embedder = eqx.nn.Embedding(
            num_embeddings=max_length, embedding_size=embedding_size, key=position_key
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        token_ids: Array,
        position_ids: Array,
        segment_ids: Array,
        enable_dropout: bool = False,
        key: jax.random.PRNGKey = None,
    ) -> Array:
        tokens = jax.vmap(self.token_embedder)(token_ids)
        segments = jax.vmap(self.segment_embedder)(segment_ids)
        positions = jax.vmap(self.position_embedder)(position_ids)
        embedded_inputs = tokens + segments + positions
        embedded_inputs = jax.vmap(self.layernorm)(embedded_inputs)
        embedded_inputs = self.dropout(
            embedded_inputs, inference=not enable_dropout, key=key
        )
        return embedded_inputs
    

class Encoder(eqx.Module):
    """Full BERT encoder."""

    embedder_block: EmbedderBlock
    layers: list[TransformerLayer]
    pooler: eqx.nn.Linear

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        type_vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        embedder_key, layer_key, pooler_key = jax.random.split(key, num=3)
        self.embedder_block = EmbedderBlock(
            vocab_size=vocab_size,
            max_length=max_length,
            type_vocab_size=type_vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=embedder_key,
        )

        layer_keys = jax.random.split(layer_key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    key=layer_key,
                )
            )

        self.pooler = eqx.nn.Linear(
            in_features=hidden_size, out_features=hidden_size, key=pooler_key
        )

    def __call__(
        self,
        token_ids: Array,
        position_ids: Array,
        segment_ids: Array,
        *,
        enable_dropout: bool = False,
        key: jax.random.PRNGKey = None,
    ) -> dict[str, Array]:
        emb_key, l_key = (None, None) if key is None else jax.random.split(key)

        embeddings = self.embedder_block(
            token_ids=token_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            enable_dropout=enable_dropout,
            key=emb_key,
        )

        # We assume that all 0-values should be masked out.
        mask = jnp.asarray(token_ids != 0, dtype=jnp.int32)

        x = embeddings
        layer_outputs = []
        for layer in self.layers:
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key)
            x = layer(x, mask, enable_dropout=enable_dropout, key=cl_key)
            layer_outputs.append(x)

        # BERT pooling.
        # The first token in the last layer is the embedding of the "[CLS]" token.
        first_token_last_layer = x[..., 0, :]
        pooled = self.pooler(first_token_last_layer)
        pooled = jnp.tanh(pooled)

        return {"embeddings": embeddings, "layers": layer_outputs, "pooled": pooled}
    

class BertClassifier(eqx.Module):
    """BERT classifier."""

    encoder: Encoder
    classifier_head: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: Mapping, num_classes: int, key: jax.random.PRNGKey):
        encoder_key, head_key = jax.random.split(key)

        self.encoder = Encoder(
            vocab_size=config["vocab_size"],
            max_length=config["max_position_embeddings"],
            type_vocab_size=config["type_vocab_size"],
            embedding_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            dropout_rate=config["hidden_dropout_prob"],
            attention_dropout_rate=config["attention_probs_dropout_prob"],
            key=encoder_key,
        )
        self.classifier_head = eqx.nn.Linear(
            in_features=config["hidden_size"], out_features=num_classes, key=head_key
        )
        self.dropout = eqx.nn.Dropout(config["hidden_dropout_prob"])

    def __call__(
        self,
        inputs: dict[str, Array],
        enable_dropout: bool = True,
        key: jax.random.PRNGKey = None,
    ) -> Array:
        seq_len = inputs["token_ids"].shape[-1]
        position_ids = jnp.arange(seq_len)

        e_key, d_key = (None, None) if key is None else jax.random.split(key)

        pooled_output = self.encoder(
            token_ids=inputs["token_ids"],
            segment_ids=inputs["segment_ids"],
            position_ids=position_ids,
            enable_dropout=enable_dropout,
            key=e_key,
        )["pooled"]
        pooled_output = self.dropout(
            pooled_output, inference=not enable_dropout, key=d_key
        )

        return self.classifier_head(pooled_output)

def compute_loss(classifier, inputs, key):
    batch_size = inputs["token_ids"].shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    logits = jax.vmap(classifier, in_axes=(0, None, 0))(inputs, True, batched_keys)
    # all of these operations are done in full precision
    return mpx.force_full_precision(jnp.mean)(
        mpx.force_full_precision(optax.softmax_cross_entropy_with_integer_labels)(
            logits=logits, labels=inputs["label"]
        )
    )


def make_step(model, inputs, opt_state, key, tx, scaling: mpx.DynamicLossScaling):
    key, new_key = jax.random.split(key)
    loss, scaling, grads_finite, grads = mpx.filter_value_and_grad(compute_loss, scaling)(model, inputs, key)
    
    model, opt_state = mpx.optimizer_update(model, tx, opt_state, grads, grads_finite)
    return loss, model, opt_state, new_key, scaling


def make_eval_step(model, inputs):
    return jax.vmap(functools.partial(model, enable_dropout=False))(inputs)
    
if __name__ == "__main__":
    # Tiny-BERT config.
    bert_config = {
        "train_mixed_precision": True,
        "vocab_size": 30522,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_act": "gelu",
        "intermediate_size": 512,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
    }

    key = jax.random.PRNGKey(5678)
    model_key, train_key = jax.random.split(key)
    classifier = BertClassifier(config=bert_config, num_classes=2, key=model_key)

    tokenizer = AutoTokenizer.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2", model_max_length=128
    )

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    ds = load_dataset("sst2")
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="jax", columns=["input_ids", "token_type_ids", "label"])

    epochs = 3
    batch_size = 32
    learning_rate = 1e-5

    ############################
    # init model
    ############################
    model = BertClassifier(config=bert_config, num_classes=2, key=model_key)

    ############################
    # init optimizer
    ############################
    tx = optax.adam(learning_rate=learning_rate)
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
    opt_state = tx.init(model)

    ############################
    # init scaling
    ############################
    if bert_config["train_mixed_precision"]:
        loss_scaling = mpx.DynamicLossScaling(loss_scaling=jnp.ones((1,), dtype=jnp.float32) * int((2 - 2**(-10)) * 2**15), 
                                          min_loss_scaling=jnp.ones((1,), dtype=jnp.float32) * 1.0)
    else:
        loss_scaling = None

    ############################
    # training 
    ############################
    for epoch in range(epochs):
        with tqdm.tqdm(
            ds["train"].iter(batch_size=batch_size, drop_last_batch=True),
            total=ds["train"].num_rows // batch_size,
            unit="steps",
            desc=f"Epoch {epoch+1}/{epochs}",
        ) as tqdm_epoch:
        
            for batch in tqdm_epoch:
                token_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
                label = batch["label"]

                # swap time and feature axis.
                token_ids = es.jax_einshape("bhn->bnh", token_ids)
                token_type_ids = es.jax_einshape("bhn->bnh", token_type_ids)

                inputs = {
                    "token_ids": token_ids,
                    "segment_ids": token_type_ids,
                    "label": label,
                }
                loss, model, opt_state, train_key, loss_scaling = make_step(
                    model, inputs, opt_state, train_key, tx, scaling=loss_scaling
                )

                tqdm_epoch.set_postfix(loss=np.sum(loss).item())

        outputs = []
        for batch in tqdm.tqdm(
            ds["validation"].iter(batch_size=batch_size),
            unit="steps",
            total=np.ceil(ds["validation"].num_rows / batch_size),
            desc="Validation",
        ):
            token_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
            label = batch["label"]


            inputs = {"token_ids": token_ids, "segment_ids": token_type_ids}

            # Compare predicted class with label.
            output = make_eval_step(model, inputs)
            output = map(float, np.argmax(output.reshape(-1, 2), axis=-1) == label)
            outputs.extend(output)

        print(f"Accuracy: {100 * np.sum(outputs) / len(outputs):.2f}%")


    
