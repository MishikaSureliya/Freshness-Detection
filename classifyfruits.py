import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# Set the path to your dataset folder on the desktop
desktop_path = os.path.expanduser("C:/Users/mishi/OneDrive/Desktop/flipkart/dataset")

# Update the dataset paths for training and testing
train_set_path = os.path.join(desktop_path, 'train')
val_set_path = os.path.join(desktop_path, 'test')

# Data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 64
TARGET_SIZE = (224, 224)

train_generator = train_datagen.flow_from_directory(train_set_path, target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE, class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_set_path, target_size=TARGET_SIZE,
                                                batch_size=BATCH_SIZE, class_mode='categorical')

# Build the MobileNetV2 model
input_shape = (224, 224, 3)
mobilenet_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
mobilenet_model.trainable = True

# Add classification layers
input_layer = tf.keras.layers.Input(shape=input_shape)
mobilenet_output = mobilenet_model(input_layer)
x = GlobalAveragePooling2D()(mobilenet_output)
x = Dense(units=6, activation='softmax')(x)

# Create final model
model = Model(inputs=input_layer, outputs=x)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5), metrics=['accuracy'])

# Train the model
EPOCHS = 4
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=val_generator,
                    validation_steps=val_generator.n // BATCH_SIZE,
                    verbose=1)

# Save the model
model.save('Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5')

# Plot training history (accuracy and loss)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('MobileNetV2 Training Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1, EPOCHS + 1))

# Accuracy plot
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, EPOCHS + 1, 1))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch #')
ax1.set_title('Accuracy')
ax1.legend(loc="best")

# Loss plot
ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, EPOCHS + 1, 1))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch #')
ax2.set_title('Loss')
ax2.legend(loc="best")

plt.show()

# Evaluate the model on validation data
model.evaluate(val_generator, steps=val_generator.n // BATCH_SIZE)