import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset import create_generators
from model import build_model

def train_model(root_dir, batch_size=64, epochs=20, output_model="emotion_model.h5"):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    train_gen, val_gen, _ = create_generators(train_dir, val_dir, "", batch_size)

    model = build_model()

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(output_model, monitor="val_accuracy", save_best_only=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen, 
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=1 
    )

    print("Training complete. Best model saved to:", output_model)

    print("\nTraining history:")
    for key in history.history.keys():
        print(f"{key}: {history.history[key]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--output_model", type=str, default="emotion_model.h5", help="Output model file name")
    args = parser.parse_args()

    train_model(args.root, args.batch_size, args.epochs, args.output_model)
