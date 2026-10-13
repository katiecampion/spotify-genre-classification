from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_classifiers():
    #models used for comparison
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        ),

        "SVM": SVC(
            kernel="rbf",
            random_state=42
        )
    }


def train_all_classifiers(X_train, y_train, X_val, y_val, verbose=True):
    classifiers = get_classifiers()
    results = {}

    for name, model in classifiers.items():
        if verbose:
            print(f"\nTraining {name}")

        #train the model on the training data
        model.fit(X_train, y_train)

        #check training and validation accuracy
        train_accuracy = model.score(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)

        results[name] = {
            "classifier": model,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        }

        if verbose:
            print(f"Training accuracy:   {train_accuracy:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")

    return results


def compare_classifiers(results):
    comparison = []

    for name, result in results.items():
        comparison.append({
            "name": name,
            "train_accuracy": result["train_accuracy"],
            "val_accuracy": result["val_accuracy"]
        })

    comparison.sort(
        key=lambda item: item["val_accuracy"],
        reverse=True
    )

    return comparison


if __name__ == "__main__":
    print("This file contains the comparison classifiers.")