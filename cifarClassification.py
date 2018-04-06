from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
import tensorflow as tf

LOGDIR = "/tmp/IntelAIworkshop/"
nbEpochs = int(input('How many epochs of training would you like to run? '))
batchSize = int(input('How many samples would you like to train on before updating the weights? '))
modelSave = bool(input('Would you like to save your model after training? Leave blank and press enter if you do not wish to. '))


class TB(tf.keras.callbacks.TensorBoard):
    #  code posted by VladislavZavadskyy at https://github.com/keras-team/keras/issues/6692
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

# Define the model: 4 convolutional layers, 2 max pools
model = tf.keras.Sequential()

with tf.variable_scope('conv1'):
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=training_scaled.shape[1:],
                     strides=(1, 1),
                     # kernel_initializer='TruncatedNormal',
                     name='conv1'))
    model.add(Activation('relu'))

with tf.variable_scope('conv2'):
    model.add(Conv2D(32, (3, 3), padding='same',
                     strides=(1, 1),
                     # kernel_initializer='TruncatedNormal',
                     name='conv2'))
    model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
model.add(Dropout(0.25, name='dropout1'))

with tf.variable_scope('conv3'):
    model.add(Conv2D(64, (3, 3), padding='same',
                     strides=(1, 1),
                     # kernel_initializer='TruncatedNormal',
                     name='conv3'))
    model.add(Activation('relu'))

with tf.variable_scope('conv4'):
    model.add(Conv2D(64, (3, 3), padding='same',
                     strides=(1, 1),
                     # kernel_initializer='TruncatedNormal',
                     name='conv4'))
    model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

model.add(Flatten(name='flat1'))  # converts feature maps to feature vectors

with tf.variable_scope('fc1'):
    model.add(Dense(1024,
                    # kernel_initializer='TruncatedNormal',
                    name='fc1'))
    model.add(Activation('relu'))

model.add(Dropout(0.5, name='dropout2'))  # reset half of the weights to zero

with tf.variable_scope('output'):
    model.add(Dense(10,
                    name='output'))
    model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tensorboard = TB(log_dir=LOGDIR,
                 histogram_freq=1,
                 batch_size=20,
                 write_graph=True,
                 write_grads=False,
                 write_images=False)


print("Initializing the model...")
# fits the model on batches, waits to send the error back after the number of batches
model.fit(training_scaled,
          labels_train,
          batch_size=batchSize,
          epochs=nbEpochs,
          validation_data=(test_scaled, labels_test),
          verbose=1,
          callbacks=[tensorboard])

if modelSave:
    model.save('./cifarClassification.h5')

print('To run tensorboard, open up either Firefox or Chrome and type localhost:6006 in the address bar.')
print('Then run `tensorboard --logdir=%s` in your terminal.' % LOGDIR)
print('Running on mac? Provide this flag: '
      '--host=localhost to the previous terminal string.')
