from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_promoter_data():
    dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
    )

    # filter promoter task
    full_train = dataset["train"].filter(
        lambda x: x["task"] == "promoter_all"
    )

    test = dataset["test"].filter(
        lambda x: x["task"] == "promoter_all"
    )

    # create train/validation split manually
    train_indices, val_indices = train_test_split(
        range(len(full_train)),
        test_size=0.15,
        random_state=42,
        shuffle=True
    )

    train = full_train.select(train_indices)
    val   = full_train.select(val_indices)

    return train, val, test
