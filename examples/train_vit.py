import itertools
import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import optax
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
import equinox as eqx

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray 

import tensorflow as tf
import tensorflow_datasets as tfds

from examples.vit import VIT

import mpx

tf.config.set_visible_devices([], 'GPU')


def filtered_device_put(tree, sharding):
    dynamic, static = eqx.partition(tree, eqx.is_array)
    dynamic = jax.device_put(dynamic, sharding)
    return eqx.combine(dynamic, static)


def predict_batch(model: eqx.Module, 
                  batch: dict,
                  inference: bool, 
                  key: PRNGKeyArray) -> Array:
    subkeys = jax.random.split(key, len(batch["input"]))
    pred = jax.vmap(model, (0, None, 0))(batch["input"], inference, subkeys)
    return pred


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


def train_epoch(model: eqx.Module, 
                optimizer: any,
                optimizer_state: PyTree, 
                train_dataset: tf.data.Dataset, 
                num_batches: Int,
                batch_sharding: jax.sharding.NamedSharding,
                replicated_sharding: jax.sharding.NamedSharding,
                loss_scaling: mpx.DynamicLossScaling | None,
                key: PRNGKeyArray,
                train_mixed_precicion: bool,
                show_progress: bool) -> tuple[eqx.Module, PyTree, PRNGKeyArray]:
    loss_value = 0
    num_datapoints = 0
    
    for batch in tqdm(itertools.islice(train_dataset, num_batches), disable=not show_progress and False):
        batch = jax.device_put(batch, batch_sharding)

        key, subkey = jax.random.split(key)

        loss_batch = 0
        model, optimizer_state, loss_scaling, loss_batch = make_step(model=model, 
                                                    optimizer=optimizer, 
                                                    optimizer_state=optimizer_state,
                                                    batch=batch,
                                                    batch_sharding=batch_sharding,
                                                    replicated_sharding=replicated_sharding,
                                                    loss_scaling=loss_scaling,
                                                    weight_regularization=config["weight_regularization"],
                                                    key=subkey,
                                                    train_mixed_precicion=train_mixed_precicion
                                                    )
        loss_value += loss_batch.astype(jnp.float32) * len(batch["input"])
        num_datapoints += len(batch["input"])
    
    return model, optimizer_state, loss_scaling, (loss_value) / (num_datapoints)


def eval_epoch(model: eqx.Module,
             val_dataset: tf.data.Dataset,
             num_batches: Int,
             batch_sharding: jax.sharding.NamedSharding,
             replicated_sharding: jax.sharding.NamedSharding,
             loss_scaling: mpx.DynamicLossScaling | None,
             key: PRNGKeyArray,
             train_mixed_precicion: bool,
             show_progress: bool) -> Float:
    loss_value = 0
    acc = 0
    num_datapoints = 0

    for batch in tqdm(itertools.islice(val_dataset, num_batches), disable=not show_progress):
        batch = jax.device_put(batch, batch_sharding)
        key, subkey = jax.random.split(key)
        if train_mixed_precicion:
            model, batch = mpx.cast_to_float16((model, batch))
        loss_temp, acc_temp = batched_loss_acc_wrapper(model, batch, batch_sharding, replicated_sharding, subkey)
        loss_value += loss_temp.astype(jnp.float32) * len(batch["input"])
        acc += acc_temp * len(batch["input"])
        num_datapoints += len(batch["input"])
    
    return (loss_value) / (num_datapoints), (acc) / (num_datapoints)

def main(config):
    ########################################
    # Set random seed
    ########################################
    key = jax.random.PRNGKey(0) 

    ########################################
    # Load dataset
    ########################################
    length_train = 50000
    length_val = 10000
    train_data_source = tfds.load(f"cifar100", split="train")
    val_data_source = tfds.load(f"cifar100", split="test")

    def init_tf_dataloader_image(data_source, batch_size, num_epochs, seed, resolution, num_parallel_calls=tf.data.AUTOTUNE):
        data = data_source.shuffle(10000, seed=seed)
        data = data.map(lambda x: {"input": tf.image.resize(x["image"], (resolution, resolution)), "target": x["label"]}, num_parallel_calls=num_parallel_calls)
        data = data.map(lambda x: {"input": x["input"] / 255, "target": x["target"]}, num_parallel_calls=num_parallel_calls)
        data = data.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.repeat(num_epochs)
        data = data.prefetch(1)
        data = data.as_numpy_iterator()
        return data

    train_dataset = init_tf_dataloader_image(train_data_source, config["batch_size"], config["num_epochs"], 0, 32)
    val_dataset = init_tf_dataloader_image(val_data_source, config["batch_size"], config["num_epochs"], 0, 32)

    #########################################
    # Sharding
    #########################################
    devices = jax.devices(backend="gpu")
    print(f"Found devices: {devices}")
    mesh = jax.make_mesh((len(devices), ), ("batch",), devices=devices)
    batch_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())

    ########################################
    # Load model
    ########################################
    key, subkey = jax.random.split(key)
    model = VIT(output_dim=100,
                num_features=config["num_features"],
                num_heads=config["num_heads"],
                num_features_residual=config["num_features_residual"],
                num_transformer_blocks=config["num_transformer_blocks"],
                num_features_head=100,
                dropout_rate=0.1,
                key=subkey)


    model = filtered_device_put(model, replicated_sharding)

    if config["train_mixed_precision"]:
        loss_scaling = mpx.DynamicLossScaling(loss_scaling=jnp.ones((1,), dtype=jnp.float32) * int((2 - 2**(-10)) * 2**15), min_loss_scaling=jnp.ones((1,), dtype=jnp.float32) * 1.0, period=2000)
        loss_scaling = filtered_device_put(loss_scaling, replicated_sharding)
    else:
        loss_scaling = None

    ########################################
    # Load optimizer
    ########################################
    # optimizer strategy from https://arxiv.org/abs/2106.10270
    duration_linear_schedule = 1000
    linear_schedule = optax.linear_schedule(
        init_value=config["learning_rate"] * 0.01,
        end_value=config["learning_rate"],
        transition_steps=duration_linear_schedule,
    )
    duration_cosine_schedule = config["num_epochs"] * int(length_train / config["batch_size"] - 1e-5) - duration_linear_schedule
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=config["learning_rate"],
        decay_steps=duration_cosine_schedule,
        alpha=0.0,
    )
    learning_rate_schedule = optax.join_schedules(
        schedules=[linear_schedule, cosine_schedule],
        boundaries=[duration_linear_schedule],
    )

    clip_transform = optax.clip_by_global_norm(1.0)
    adam_transform = optax.scale_by_adam()
    learning_rate_transform = optax.scale_by_learning_rate(learning_rate_schedule)

    optimizer = optax.chain(clip_transform, adam_transform, learning_rate_transform)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))
    optimizer_state = filtered_device_put(optimizer_state, replicated_sharding)

    ########################################
    # init logger
    ########################################
    train_losses = []
    val_losses = []
    val_accs = []
    loss_scalings = []

    ########################################
    # Train model
    ########################################
    best_val_loss = 1e6
    for epoch in range(config["num_epochs"]):
        # train
        model, optimizer_state, loss_scaling, train_loss = train_epoch(model=model, 
                                      optimizer=optimizer, 
                                      optimizer_state=optimizer_state, 
                                      train_dataset=train_dataset, 
                                      num_batches=int(length_train / config["batch_size"] - 1e-5),
                                      batch_sharding=batch_sharding,
                                      replicated_sharding=replicated_sharding,
                                      loss_scaling=loss_scaling,
                                      key=key, 
                                      train_mixed_precicion=config["train_mixed_precision"],
                                      show_progress=True)
        if loss_scaling is not None:
            loss_scalings.append(loss_scaling.loss_scaling)
        train_losses.append(train_loss)
        
        # evaluate
        val_loss, acc = eval_epoch(model=model, 
                       val_dataset=val_dataset,
                       num_batches=int(length_val / config["batch_size"] - 1e-5),
                       batch_sharding=batch_sharding,
                       replicated_sharding=replicated_sharding,
                       loss_scaling=loss_scaling,
                       key=key, 
                       train_mixed_precicion=config["train_mixed_precision"],
                       show_progress=True)
        
        val_losses.append(val_loss)
        val_accs.append(acc*100)

        print(f"Epoch {epoch}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {acc*100}")


    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = {
        "train_mixed_precision": True,
        "batch_size": 512,
        "num_epochs": 10,
        "num_features": 128,
        "num_heads": 4,
        "num_features_residual": 256,
        "num_transformer_blocks": 12,
        "learning_rate": 0.001,
        "batch_size": 128,
        "weight_regularization": 0.001,
    }
    main(config)
