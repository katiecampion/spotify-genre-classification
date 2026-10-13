import os
import warnings
warnings.filterwarnings("ignore")

from data_loader import prepare_data, SELECTED_GENRES, AUDIO_FEATURES
from eda import run_eda
from models.neural_network import create_network
from models.classifiers import train_all_classifiers
from evaluation import (
    evaluate_model,
    plot_training_curves,
    plot_model_comparison,
    create_results_summary
)
from utils import ensure_dir, set_random_seed, Timer, print_data_summary

RANDOM_SEED = 42
RESULTS_DIR = "results"
DATASET_PATH = "dataset.csv"

NN_HIDDEN_LAYERS = [32, 16]
NN_LEARNING_RATE = 0.01
NN_EPOCHS = 100
NN_BATCH_SIZE = 32


def prepare_dataset():
    #prepare the dataset
    data = prepare_data(
        filepath=DATASET_PATH,
        genres=SELECTED_GENRES,
        features=AUDIO_FEATURES
    )
    #print summary of the prepared data
    print_data_summary(data)

    return data

def run_eda_analysis():

    #explore the dataset
    run_eda(
        filepath=DATASET_PATH,
        genres=SELECTED_GENRES
    )


def train_neural_network(data):
    #train the neural network
    nn = create_network(
        n_features=data["n_features"],
        n_classes=data["n_classes"],
        hidden_layers=NN_HIDDEN_LAYERS,
        learning_rate=NN_LEARNING_RATE
    )

    print("Network architecture:")
    print([data["n_features"]] + NN_HIDDEN_LAYERS + [data["n_classes"]])

    with Timer("Neural network training"):
        history = nn.fit(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            verbose=True
        )

    plot_training_curves(
        history,
        save_path=f"{RESULTS_DIR}/nn_training_curves.png"
    )

    print("\nNeural Network Accuracy")
    print(f"Training:   {nn.score(data['X_train'], data['y_train']):.4f}")
    print(f"Validation: {nn.score(data['X_val'], data['y_val']):.4f}")
    print(f"Test:       {nn.score(data['X_test'], data['y_test']):.4f}")

    return nn, history


def train_classifiers(data):
    print("\nTraining comparison classifiers:")

    classifiers = train_all_classifiers(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        verbose=True
    )

    return classifiers


def evaluate_all_models(data, nn, classifiers):
    print("\nEvaluating models on test set:")

    all_results = {}

    #evaluate neural network
    nn_metrics = evaluate_model(
        nn,
        data["X_test"],
        data["y_test"],
        data["class_names"],
        model_name="Neural Network NumPy"
    )

    all_results["Neural Network NumPy"] = nn_metrics

    #evaluate comparison models
    for name, result in classifiers.items():
        model = result["classifier"]

        metrics = evaluate_model(
            model,
            data["X_test"],
            data["y_test"],
            data["class_names"],
            model_name=name
        )

        all_results[name] = metrics

    comparison_data = {
        name: {"val_accuracy": metrics["accuracy"]}
        for name, metrics in all_results.items()
    }

    plot_model_comparison(
        comparison_data,
        save_path=f"{RESULTS_DIR}/model_comparison.png"
    )

    create_results_summary(
        all_results,
        save_path=f"{RESULTS_DIR}/results_summary.txt"
    )

    return all_results

def main():
    print("Music Genre Classification Project")


    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    set_random_seed(RANDOM_SEED)


    ensure_dir(RESULTS_DIR)

    data = prepare_dataset()

    run_eda_analysis()

    nn, history = train_neural_network(data)

    classifiers = train_classifiers(data)


    evaluate_all_models(data, nn, classifiers)

    print("\nDone")


if __name__ == "__main__":
    main()