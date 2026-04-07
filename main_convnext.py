import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import nnx
from orbax.export import ExportManager, JaxModule, ServingConfig

# Add bonsai to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bonsai'))
from bonsai.models.convnext.modeling import ConvNeXt, ModelConfig

# Print JAX backend info
print("JAX platform:", jax.default_backend())
print("JAX devices:", jax.devices())

# ---- Hyperparameters ----
train_steps = 1200
eval_every = 200
batch_size = 32
learning_rate = 0.005
momentum = 0.9

# ---- Load and preprocess MNIST ----
tf.random.set_seed(0)

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
    lambda sample: {
        'image': tf.cast(sample['image'], tf.float32) / 255,
        'label': sample['label'],
    }
)
test_ds = test_ds.map(
    lambda sample: {
        'image': tf.cast(sample['image'], tf.float32) / 255,
        'label': sample['label'],
    }
)

train_ds = train_ds.repeat().shuffle(1024)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)


# ---- Define the ConvNeXt model config for MNIST ----
# Use a small custom config suitable for 28x28 grayscale images.
# With patch_size=(4,4): 28/4 = 7x7 feature map after embedding.
# Stages downsample: 7 -> 7 -> 4 -> 2 -> 1 (global avg pool).
config = ModelConfig(
    stage_depths=(1, 1, 1, 1),
    stage_dims=(32, 64, 128, 256),
    drop_path_rate=0.0,
    num_classes=10,
    in_channels=1,
    patch_size=(4, 4),
)

# Instantiate the model
model = ConvNeXt(cfg=config, rngs=nnx.Rngs(0))

# Verify the model runs
key = jax.random.key(0)
y = model(jnp.ones((1, 28, 28, 1)), rngs=key, train=False)
print("Model output shape:", y.shape)

# ---- Create optimizer and metrics ----
optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate, momentum), wrt=nnx.Param
)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
)


# ---- Define training and evaluation steps ----
def loss_fn(model: ConvNeXt, rng_key: jax.Array, batch, train: bool):
    logits = model(batch['image'], rngs=rng_key, train=train)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: ConvNeXt, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, rng_key: jax.Array, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, rng_key, batch, True)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: ConvNeXt, metrics: nnx.MultiMetric, rng_key: jax.Array, batch):
    loss, logits = loss_fn(model, rng_key, batch, False)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])


# ---- Train and evaluate ----
metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
}

key = jax.random.key(0)

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    key, step_key = jax.random.split(key)
    train_step(model, optimizer, metrics, step_key, batch)

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        for metric, value in metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)
        metrics.reset()

        for test_batch in test_ds.as_numpy_iterator():
            key, eval_key = jax.random.split(key)
            eval_step(model, metrics, eval_key, test_batch)

        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
        metrics.reset()

        print(
            f"Step {step}: "
            f"train_loss={metrics_history['train_loss'][-1]:.4f}, "
            f"train_acc={metrics_history['train_accuracy'][-1]:.4f}, "
            f"test_loss={metrics_history['test_loss'][-1]:.4f}, "
            f"test_acc={metrics_history['test_accuracy'][-1]:.4f}"
        )

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train', 'test'):
    ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
    ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.savefig('training_curves_convnext.png', dpi=150, bbox_inches='tight')
print("Saved training curves to training_curves_convnext.png")

# ---- Inference on test set ----
@nnx.jit
def pred_step(model: ConvNeXt, batch):
    logits = model(batch['image'], rngs=jax.random.key(0), train=False)
    return logits.argmax(axis=1)


test_batch = next(test_ds.as_numpy_iterator())
pred = pred_step(model, test_batch)

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
    ax.set_title(f'label={pred[i]}')
    ax.axis('off')
plt.savefig('predictions_convnext.png', dpi=150, bbox_inches='tight')
print("Saved predictions to predictions_convnext.png")

# ---- Export the model ----
def exported_predict(model, y):
    return model(y, rngs=jax.random.key(0), train=False)


jax_module = JaxModule(model, exported_predict)
sig = [tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.float32)]

os.makedirs('saved_models', exist_ok=True)
export_mgr = ExportManager(jax_module, [
    ServingConfig('mnist_convnext_server', input_signature=sig)
])

output_dir = 'saved_models/demo_convnext_model'
export_mgr.save(output_dir)
print(f"Model exported to {output_dir}/")
print("Contents:", os.listdir(output_dir))
