import os
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.optimizers import Adam

# Paths (configurable via env; defaults to repository layout)
BASE = Path(__file__).parent
TRAIN_PATH = Path(os.getenv("DATASET_DIR", BASE / "dataset"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE / "model"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "mask_detector.h5"

print(f"Dataset dir: {TRAIN_PATH}")
print(f"Model will be saved to: {MODEL_PATH}")

# Data generators with MobileNetV2 preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

target_size = (224, 224)
batch_size = 32

train_gen = datagen.flow_from_directory(
    str(TRAIN_PATH),
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = datagen.flow_from_directory(
    str(TRAIN_PATH),
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

num_classes = len(train_gen.class_indices)
print("Class indices:", train_gen.class_indices)
print("Detected classes:", num_classes)
if num_classes != 3:
    raise RuntimeError("Dataset must contain exactly three folders: 'with_mask', 'without_mask', and 'improper_mask'")

# Base model: MobileNetV2 (transfer learning)
base = MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base.trainable = False  # freeze base for warmup

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base.input, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
checkpoint = ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True)

# Phase 1: warmup
warmup_epochs = int(os.getenv("WARMUP_EPOCHS", "5"))
model.fit(train_gen, epochs=warmup_epochs, validation_data=val_gen, callbacks=[early_stop, checkpoint])

# Phase 2: fine-tune last blocks
for layer in base.layers[-40:]:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

epochs = int(os.getenv("EPOCHS", "20"))
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[early_stop, checkpoint])

model.save(str(MODEL_PATH))
print(f"Model saved to: {MODEL_PATH}")
print("To convert to TFJS (run in shell with tfjs converter installed):")
print("tensorflowjs_converter --input_format=keras " + str(MODEL_PATH) + " " + str(BASE.parent / "frontend" / "public" / "model"))
