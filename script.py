import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tfgan = tf.contrib.gan

y = np.random.normal(size=(100000, 1))

BATCH_SIZE = 1000
model_dir = "models/test0"


# See TFGAN's `train.py` for a description of the generator and
# discriminator API.
def generator_fn(*args):
  print("generator:")
  for a in args:
      print(a)
  print("")

  generator_inputs = args[0] ['x']
  hidden_layer = generator_inputs
  for i in range(4):
      hidden_layer = tf.layers.dense(hidden_layer, units=5, activation=tf.nn.relu)
  return tf.layers.dense(hidden_layer, units=1, activation=None)

def discriminator_fn(*args):
  print("discriminator:")
  for a in args:
      print(a)
  print("")

  data = args[0]
  hidden_layer = data
  for i in range(5):
      hidden_layer = tf.layers.dense(hidden_layer, units=5, activation=tf.nn.relu)
  return tf.layers.dense(hidden_layer, units=1, activation=None)


# Create GAN estimator.
gan_estimator = tfgan.estimator.GANEstimator(
    model_dir,
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=lambda *args, **kwargs: tfgan.losses.wasserstein_discriminator_loss(*args, **kwargs) + \
                                                  tfgan.losses.wasserstein_gradient_penalty(*args, **kwargs),
    generator_optimizer=tf.train.AdamOptimizer(0.001),#(0.1, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.001))#(0.1, 0.5))


# Train estimator.
gan_estimator.train(
    lambda: ({'x' : tf.random_uniform((BATCH_SIZE, 3), dtype='float32')},
             tf.data.Dataset.from_tensor_slices(tf.cast(y, dtype='float32')).repeat(100) \
                    .batch(BATCH_SIZE).make_one_shot_iterator().get_next()))

# Evaluate resulting estimator.
print(gan_estimator.evaluate(
    lambda: ({'x' : tf.random_uniform((100000, 3), dtype='float32')},
             tf.data.Dataset.from_tensor_slices(tf.cast(y, dtype='float32')) \
                    .batch(100000).make_one_shot_iterator().get_next())))


print("now predicting")
# Generate samples from generator.
predictions = np.array([
    x for x in gan_estimator.predict(
        lambda: {'x' : tf.data.Dataset.from_tensor_slices(
                            tf.cast(np.random.uniform(size=(100000, 3)), dtype='float32')
                       ).batch(100000).make_one_shot_iterator().get_next()})])
print("done")


_, bins, _ = plt.hist(y, bins=300, label='real')
_, _   , _ = plt.hist(predictions, bins=bins, label='generated')
plt.legend()
plt.show()