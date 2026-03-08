from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_yawn"
MODEL_PATH = BASE_DIR / "models" / "yawn_cnn.h5"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
valid_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    DATA_DIR / "train",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
)

valid_data = valid_gen.flow_from_directory(
    DATA_DIR / "valid",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
)

model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_loss"),
]

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(MODEL_PATH)
print(f"Saved yawn model to: {MODEL_PATH.resolve()}")
print(f"Training epochs run: {len(history.history['loss'])}")
print(f"Class mapping: {train_data.class_indices}")
