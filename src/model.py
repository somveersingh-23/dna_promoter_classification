from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_cnn(input_shape):
    model = Sequential()

    model.add(Conv1D(64, kernel_size=8, activation="relu",
                     input_shape=input_shape))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, kernel_size=6, activation="relu"))
    model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
