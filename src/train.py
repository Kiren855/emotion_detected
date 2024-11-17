import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset import create_generators
from model import build_model
import tensorflow as tf 

def train_model(root_dir, batch_size=64, epochs=20, output_model="emotion_model.h5"):
    result_folder = "result"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    output_model_path = os.path.join(result_folder, output_model)
    history_csv_path = os.path.join(result_folder, "training_history.csv")

    train_dir = os.path.join(root_dir, "train")
    
    train_gen, val_gen, _ = create_generators(train_dir, None, batch_size)

    model = build_model()
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  metrics=['accuracy'])
        
    checkpoint = ModelCheckpoint(output_model_path, monitor="val_accuracy", save_best_only=True)
    
    history = model.fit(
                train_gen,
                steps_per_epoch=22968 // 64,
                validation_data=val_gen,
                validation_steps=5741 // 64, 
                epochs=epochs, 
                callbacks=[checkpoint],
                verbose=1
            )
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv_path, index=False)
    
    print("Training complete. Best model saved to:", output_model_path)
    print("\nTraining history saved to:", history_csv_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--output_model", type=str, default="emotion_model.h5", help="Output model file name")
    args = parser.parse_args()

    train_model(args.root, args.batch_size, args.epochs, args.output_model)
