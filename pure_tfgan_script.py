### based on an example from:
### https://github.com/tensorflow/tensorflow/tree/r1.11/tensorflow/contrib/gan

import io

import PIL

import numpy as np
import tensorflow as tf
tfgan = tf.contrib.gan

import matplotlib.pyplot as plt

gpu_options = tf.GPUOptions(allow_growth=True)
tf_config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)
#config = tf.estimator.RunConfig(session_config=tf_config)



BATCH_SIZE = 1000

# Set up the input.
y = tf.data.Dataset.from_tensor_slices(np.random.normal(size=(100000, 1)).astype('float32')) \
        .repeat().batch(BATCH_SIZE).make_one_shot_iterator().get_next()
#noise = tf.random_uniform([BATCH_SIZE, 1], dtype='float32')
noise = tf.random_normal([BATCH_SIZE, 1], mean=5., stddev=8., dtype='float32')


def generator_fn(gen_input):
    hidden_layer = gen_input
    for _ in range(1):
        hidden_layer = tf.layers.dense(hidden_layer, units=100, activation=tf.nn.relu)
    result = tf.layers.dense(hidden_layer, units=1, activation=None)

    tf.summary.histogram('generator/prediction', result)
    return result

def discriminator_fn(disc_input, _):
    hidden_layer = disc_input
    for _ in range(2):
        hidden_layer = tf.layers.dense(hidden_layer, units=100, activation=tf.nn.relu)
    return tf.layers.dense(hidden_layer, units=1, activation=None)


# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=generator_fn,  # you define
    discriminator_fn=discriminator_fn,  # you define
    real_data=y,
    generator_inputs=noise)

# Predictions for evaluation
with tf.variable_scope('Generator', reuse=True):
    #preds_tensor = gan_model.generator_fn(tf.random_uniform([100000, 1], dtype='float32'))
    preds_tensor = gan_model.generator_fn(tf.random_normal([100000, 1], mean=5., stddev=8., dtype='float32'))

# Build the GAN loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty_weight=10.0)



# Create the train ops, which calculate gradients and apply updates to weights.
train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5))
    #generator_optimizer=tf.train.AdamOptimizer(0.01, 0.5),
    #discriminator_optimizer=tf.train.AdamOptimizer(0.05, 0.5))


train_hooks_fn = tfgan.get_sequential_train_hooks(
    tfgan.GANTrainSteps(
        generator_train_steps=1,
        discriminator_train_steps=10))

class RunOpsEveryNSteps(tf.train.SessionRunHook):
    """Hook that runs given ops every N steps"""

    def __init__(self, num_steps, run_ops_fn, log=True, title=None):
        self._num_steps = num_steps
        self._run_ops_fn = run_ops_fn
        self._log = log
        if title is None:
            self._title = 'RunOpEvery_{}_Steps'.format(num_steps)
        else:
            self._title = title
    
    def begin(self):
        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._run_ops = self._run_ops_fn()

    def after_run(self, run_context, run_values):
        gstep = run_context.session.run(self._global_step_tensor)
        if gstep % self._num_steps == 0:
            res = {k: run_context.session.run(v) for k, v in self._run_ops.items()}
            if self._log:
                print("{}, Step #{}:".format(self._title, gstep))
                for k, v in res.items():
                    print("  {} : {}".format(k, v))


def BuildEvalOp():
    def PlotGeneratedHist(x):
        fig = plt.figure(figsize=(10, 10))
        plt.hist(x, bins=int(np.ceil(len(x)**0.5)))
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        img = PIL.Image.open(buf)
        return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)

    fig = tf.py_func(PlotGeneratedHist, [preds_tensor], tf.uint8)
    return tf.summary.image("EvalImg", fig)



hooks = train_hooks_fn(train_ops) + [tf.train.StopAtStepHook(num_steps=10000),
                                     RunOpsEveryNSteps(10,
                                                       lambda: {'gen_loss'  : gan_loss[0],
                                                                'disc_loss' : gan_loss[1]}),
                                     RunOpsEveryNSteps(10,
                                                       lambda: {'img_sum_op' : BuildEvalOp()},
                                                       log=False)]

# Run the train ops in the alternating training scheme.
tfgan.gan_train(
    train_ops,
    hooks=hooks,
    logdir="models/tfgan_test1",
    config=tf_config)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    predictions = sess.run(preds_tensor)


_, bins, _ = plt.hist(np.random.normal(size=(100000, 1)).astype('float32').flatten(), bins=300, label='real', alpha=0.8)
_, _   , _ = plt.hist(predictions, bins=bins, label='gen', alpha=0.6)
plt.legend()
plt.show()

print('ok')