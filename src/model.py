import json

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    EYE_DATA_DIR,
    EYE_MODEL_PATH,
    IMG_SIZE_EYES,
    IMG_SIZE_YAWNS,
    MODELS_DIR,
    YAWN_CLASS_MAP_PATH,
    YAWN_DATA_DIR,
    YAWN_MODEL_PATH,
)


def build_model(input_shape: tuple[int, int, int]) -> Sequential:
    """Builds a simple, robust CNN architecture suitable for both eyes and yawns."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
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
    return model


def main() -> None:
    args = get_args()

    if args.target == "eyes":
        data_dir = EYE_DATA_DIR
        img_size = IMG_SIZE_EYES
        model_path = EYE_MODEL_PATH
        use_augmentation = False
    else:
        # yawns
        data_dir = YAWN_DATA_DIR
        img_size = IMG_SIZE_YAWNS
        model_path = YAWN_MODEL_PATH
        use_augmentation = True

    if not (data_dir / "train").exists():
        print(f"[Error] Training data not found at {data_dir}. Did you run prepare_dataset.py?")
        return

    print(f"\n--- Training Model for: {args.target.upper()} ---")

    # 1. Prepare Data Generators
    if use_augmentation:
        train_gen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )
    else:
        train_gen = ImageDataGenerator(rescale=1.0 / 255)

    valid_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        data_dir / "train",
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
    )

    valid_data = valid_gen.flow_from_directory(
        data_dir / "valid",
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
    )

    # 2. Build and Train Model
    model = build_model(input_shape=(*img_size, 1))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # 3. Save Artifacts
    model.save(model_path)
    print(f"\n[Success] Model saved to: {model_path.resolve()}")
    print(f"Class mapping: {train_data.class_indices}")

    if args.target == "yawns":
        YAWN_CLASS_MAP_PATH.write_text(json.dumps(train_data.class_indices, indent=2))
        print(f"[Success] Yawn class map saved to: {YAWN_CLASS_MAP_PATH.resolve()}")


if __name__ == "__main__":
    main()
