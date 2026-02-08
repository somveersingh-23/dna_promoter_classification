from tensorflow.keras.models import load_model
from load_data import load_promoter_data
from preprocess import preprocess_dataset
from sklearn.metrics import classification_report

def evaluate():
    _, _, test_ds = load_promoter_data()
    X_test, y_test = preprocess_dataset(test_ds)

    model = load_model("models/best_model.h5")
    preds = (model.predict(X_test) > 0.5).astype(int)

    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate()
