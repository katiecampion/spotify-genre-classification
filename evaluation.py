import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    return metrics


def print_metrics(metrics):
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")


def print_report(y_true, y_pred, class_names):
    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    model_name
):
    #show which genres are being confused with each other
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(f"results/confusion_matrix_{safe_name}.png")
    plt.close()


def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #loss
    axes[0].plot(epochs, history["train_loss"], label="Training Loss")

    if len(history["val_loss"]) > 0:
        axes[0].plot(epochs, history["val_loss"], label="Validation Loss")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    #accuracy
    axes[1].plot(epochs, history["train_accuracy"], label="Training Accuracy")

    if len(history["val_accuracy"]) > 0:
        axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy")

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(results, save_path):
    names = list(results.keys())
    scores = [results[name]["val_accuracy"] for name in names]

    plt.figure(figsize=(10, 5))

    plt.bar(names, scores)

    plt.ylabel("Accuracy")
    plt.title("Model Comparison")

    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(
    model,
    X_test,
    y_test,
    class_names,
    model_name
):
    print(f"\nEvaluating {model_name}")

    #predict the classes
    y_pred = model.predict(X_test)

    #calculate evaluation metrics
    metrics = calculate_metrics(y_test, y_pred)

    print_metrics(metrics)

    #print_report(y_test, y_pred, class_names)

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        model_name
    )

    return metrics


def create_results_summary(results, save_path):
    with open(save_path, "w") as f:
        f.write("MODEL RESULTS\n")
        f.write("=" * 40 + "\n\n")

        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )

        for name, metrics in sorted_results:
            f.write(f"{name}\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    print("Evaluation module")