import json

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app_config import MODELS_DIR, YAWN_CLASS_MAP_PATH, YAWN_DATA_DIR, YAWN_MODEL_PATH

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 12


def main() -> None:
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    valid_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        YAWN_DATA_DIR / "train",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
    )

    valid_data = valid_gen.flow_from_directory(
        YAWN_DATA_DIR / "valid",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
    )

    model = Sequential(
        [
            Conv2D(16, (3, 3), activation="relu", input_shape=(64, 64, 1)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(2, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(str(YAWN_MODEL_PATH), save_best_only=True, monitor="val_loss"),
    ]

    model.fit(train_data, validation_data=valid_data, epochs=EPOCHS, callbacks=callbacks)

    model.save(YAWN_MODEL_PATH)
    YAWN_CLASS_MAP_PATH.write_text(json.dumps(train_data.class_indices, indent=2))

    print(f"Saved yawn model to: {YAWN_MODEL_PATH}")
    print(f"Saved yawn class mapping to: {YAWN_CLASS_MAP_PATH}")
    print(f"Yawn class mapping: {train_data.class_indices}")


if __name__ == "__main__":
    main()
