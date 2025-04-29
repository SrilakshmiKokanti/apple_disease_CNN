import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from utils.data_loader import get_data_loaders
from utils.model_utils import build_custom_cnn
import json

# Configs
DATA_DIR = 'dataset'
IMG_SIZE = (128, 128)        
BATCH_SIZE = 50
EPOCHS = 50
NUM_CLASSES = 4

# Load data
train_gen, val_gen, test_gen = get_data_loaders(
    data_dir=DATA_DIR, 
    batch_size=BATCH_SIZE, 
    img_size=IMG_SIZE      # Pass IMG_SIZE to ensure all generators use (128, 128)
)

# Build and compile model
model = build_custom_cnn(
    input_shape=(128, 128, 3),   # Changed from (256, 256, 3)
    num_classes=NUM_CLASSES
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Use 0.001 as a good starting point
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'save_model.h5', 
    save_best_only=True, 
    monitor='val_accuracy', 
    mode='max'
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, 
    restore_best_weights=True
)

# Train
history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=EPOCHS, 
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Evaluate on test
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save class names for app use
class_names = list(train_gen.class_indices.keys())
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
