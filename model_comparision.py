import time
import tensorflow as tf
import pandas as pd
import json
from utils.data_loader import get_data_loaders

# --- Configuration ---
DATA_DIR = 'dataset'
BATCH_SIZE = 50
EPOCHS = 5  # Reduced for faster testing; increase for actual training
results = []

# --- 1. Custom Model ---
custom_input_size = (128, 128)
# Use only test generator (no need to retrain)
_, _, test_gen = get_data_loaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    img_size=custom_input_size
)
num_classes = test_gen.num_classes
custom_model = tf.keras.models.load_model("save_model.h5")
start_time = time.time()
test_loss, test_acc = custom_model.evaluate(test_gen, verbose=0)
elapsed_time = time.time() - start_time

results.append({
    "Model": "Custom_Conv3_DCNN",
    "Accuracy": round(test_acc * 100, 2),
    "Time (s)": round(elapsed_time, 2)
})

# --- 2. Transfer Learning Models ---
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet152, DenseNet201,
    InceptionV3, Xception, MobileNet, MobileNetV2
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess

MODELS = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'ResNet152': ResNet152,
    'DenseNet201': DenseNet201,
    'InceptionV3': InceptionV3,
    'Xception': Xception,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
}

MODEL_CONFIGS = {
    'VGG16':      {'preprocess': vgg16_preprocess,       'input_size': (224, 224)},
    'VGG19':      {'preprocess': vgg19_preprocess,       'input_size': (224, 224)},
    'ResNet50':   {'preprocess': resnet_preprocess,      'input_size': (224, 224)},
    'ResNet152':  {'preprocess': resnet_preprocess,      'input_size': (224, 224)},
    'DenseNet201':{'preprocess': densenet_preprocess,    'input_size': (224, 224)},
    'InceptionV3':{'preprocess': inception_preprocess,   'input_size': (299, 299)},
    'Xception':   {'preprocess': xception_preprocess,    'input_size': (299, 299)},
    'MobileNet':  {'preprocess': mobilenet_preprocess,   'input_size': (224, 224)},
    'MobileNetV2':{'preprocess': mobilenet_v2_preprocess,'input_size': (224, 224)},
}

for name, model_fn in MODELS.items():
    try:
        print(f"\nTraining {name}...")
        config = MODEL_CONFIGS[name]
        preprocess_func = config['preprocess']
        input_size = config['input_size']

        # Data generators for this model
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_func,
            rotation_range=20,
            horizontal_flip=True,
            brightness_range=(0.8, 1.2),
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_func
        )

        train_gen = train_datagen.flow_from_directory(
            f"{DATA_DIR}/train",
            target_size=input_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        val_gen = val_test_datagen.flow_from_directory(
            f"{DATA_DIR}/val",
            target_size=input_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        test_gen = val_test_datagen.flow_from_directory(
            f"{DATA_DIR}/test",
            target_size=input_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        num_classes = train_gen.num_classes

        # Build model
        base_model = model_fn(weights='imagenet', include_top=False, input_shape=input_size + (3,))
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        start_time = time.time()
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=1)
        elapsed_time = time.time() - start_time

        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        results.append({
            "Model": name,
            "Accuracy": round(test_acc * 100, 2),
            "Time (s)": round(elapsed_time, 2)
        })
        # Save results after each model
        pd.DataFrame(results).to_csv("model_comparison_results.csv", index=False)
    except Exception as e:
        print(f"Skipping {name} due to error: {e}")

# --- Final Results ---
results_df = pd.DataFrame(results)
print("\nModel Comparison Table:\n", results_df)
results_df.to_csv("model_comparison_results.csv", index=False)
