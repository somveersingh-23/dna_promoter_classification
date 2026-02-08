from load_data import load_promoter_data
from preprocess import preprocess_dataset
from model import build_cnn

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train():
    train_ds, val_ds, test_ds = load_promoter_data()

    X_train, y_train = preprocess_dataset(train_ds)
    X_val, y_val     = preprocess_dataset(val_ds)

    model = build_cnn(input_shape=(300,4))

    checkpoint = ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[checkpoint, early_stop]
    )

if __name__ == "__main__":
    train()
