from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf

LOGDIR = "/tmp/IntelAIworkshop/"


class TB(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Max scale all the image data
training_scaled = x_train / x_train.max()
test_scaled = x_test / x_test.max()

# One hot encoding of labels
labels_train = tf.keras.utils.to_categorical(y_train, 10)
labels_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the model: 2 convolutional layers, 2 max pools
model = tf.keras.Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=training_scaled.shape[1:],
                 strides=(1, 1),
                 activation='relu',
                 # kernel_initializer='TruncatedNormal',
                 name='conv1'))

model.add(Conv2D(32, (3, 3), padding='same',
                 strides=(1, 1),
                 activation='relu',
                 # kernel_initializer='TruncatedNormal',
                 name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
model.add(Dropout(0.25, name='dropout1'))

model.add(Conv2D(64, (3, 3), padding='same',
                 strides=(1, 1),
                 activation='relu',
                 # kernel_initializer='TruncatedNormal',
                 name='conv3'))

model.add(Conv2D(64, (3, 3), padding='same',
                 strides=(1, 1),
                 activation='relu',
                 # kernel_initializer='TruncatedNormal',
                 name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

model.add(Flatten(name='flat1'))  # converts feature maps to feature vectors

model.add(Dense(256,
                activation='relu',
                # kernel_initializer='TruncatedNormal',
                name='fc1'))
model.add(Dropout(0.5, name='dropout2'))  # reset half of the weights to zero

model.add(Dense(10,
                activation='softmax',
                name='output'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tensorboard = TB(log_dir=LOGDIR,
                 histogram_freq=1,
                 batch_size=20,
                 write_graph=True,
                 write_grads=True,
                 write_images=False)


print("Initializing the model...")
# fits the model on batches, waits to send the error back after the number of batches
model.fit(training_scaled,
          labels_train,
          batch_size=20,
          epochs=5,
          validation_data=(test_scaled, labels_test),
          verbose=1,
          callbacks=[tensorboard])

model.save('cifarClassification.h5')
print('To run tensorboard, open up either Firefox or Chrome and type localhost:6006 in the address bar.')
print('Then run `tensorboard --logdir=%s` in your terminal to see the results.' % LOGDIR)
print('Running on mac? If you want to get rid of the dialogue asking to give '
      'network permissions to TensorBoard, you can provide this flag: '
      '--host=localhost')
