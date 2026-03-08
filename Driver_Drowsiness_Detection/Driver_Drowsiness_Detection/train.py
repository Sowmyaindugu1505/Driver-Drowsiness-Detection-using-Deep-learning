from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app_config import EYE_DATA_DIR, EYE_MODEL_PATH, MODELS_DIR

IMG_SIZE = (24, 24)
BATCH_SIZE = 32
EPOCHS = 12


def main() -> None:
    train_gen = ImageDataGenerator(rescale=1.0 / 255)
    valid_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        EYE_DATA_DIR / "train",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
    )

    valid_data = valid_gen.flow_from_directory(
        EYE_DATA_DIR / "valid",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
    )

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(24, 24, 1)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(str(EYE_MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]

    model.fit(train_data, validation_data=valid_data, epochs=EPOCHS, callbacks=callbacks)
    model.save(EYE_MODEL_PATH)

    print(f"Saved eye model to: {EYE_MODEL_PATH}")
    print(f"Eye class mapping: {train_data.class_indices}")


if __name__ == "__main__":
    main()
