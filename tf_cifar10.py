import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
LOGDIR = "/tmp/cifar_classifier"


def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        #  learn a set of 32 filters
        filters=32,
        #  slide a 3x3 receptive field across the input
        kernel_size=[3, 3],
        #  pad the edges of the output tensors to retain original 32x32 input shape
        padding="same",
        #  use the relu activation function
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=[2, 2])

    # Dropout Layer #1
    dropout1 = tf.layers.dropout(
        inputs=pool1,
        rate=0.25,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    # Conv Layer #3
    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Conv Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size=[2, 2],
        strides=[2, 2])

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
        inputs=dense,
        rate=0.25,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Max scale all the image data
    training_scaled = x_train / x_train.max()
    test_scaled = x_test / x_test.max()

    training_scaled = training_scaled.astype(dtype=np.float32)
    test_scaled = test_scaled.astype(dtype=np.float32)

    train_labels = np.asarray(y_train, dtype=np.int32)
    eval_labels = np.asarray(y_test, dtype=np.int32)

    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=LOGDIR)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_scaled},
        y=train_labels,
        batch_size=32,
        num_epochs=5,
        shuffle=True)

    cifar_classifier.train(
        input_fn=train_input_fn,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_scaled},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run(
        main=None,
        argv=None)

print('To run tensorboard, open up either Firefox or Chrome and type localhost:6006 in the address bar.')
print('Then run `tensorboard --logdir=%s` in your terminal.' % LOGDIR)
print('If youre on a Mac, provide the following flag: '
      '--host=localhost to the previous terminal string.')