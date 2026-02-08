import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

BASES = ["A", "T", "G", "C"]

def extract_filters():
    model = load_model("models/best_model.h5")

    # find first Conv1D layer automatically
    conv_layer = None
    for layer in model.layers:
        if "conv" in layer.name.lower():
            conv_layer = layer
            break

    if conv_layer is None:
        raise ValueError("No Conv1D layer found")

    weights, bias = conv_layer.get_weights()
    print("Conv filter shape:", weights.shape)
    return weights

def filter_to_pwm(filter_weights):
    pwm = filter_weights - filter_weights.min()
    pwm = pwm / (pwm.sum(axis=1, keepdims=True) + 1e-8)
    return pwm

def plot_motif(pwm, idx):
    plt.figure(figsize=(6,2))
    for i, base in enumerate(BASES):
        plt.plot(pwm[:, i], label=base)

    plt.title(f"Learned Motif {idx}")
    plt.xlabel("Position")
    plt.ylabel("Importance")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filters = extract_filters()

    # visualize first 3 motifs only
    for i in range(3):
        pwm = filter_to_pwm(filters[:, :, i])
        plot_motif(pwm, i)
