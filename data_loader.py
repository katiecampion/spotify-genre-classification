import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


SELECTED_GENRES = [
    "pop", "rock", "hip-hop", "electronic",
    "jazz", "classical", "metal", "country"
]

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "popularity"
]

def load_data(filepath="dataset.csv"):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def filter_genres(df):
    #keep only genres needed
    df = df[df["track_genre"].isin(SELECTED_GENRES)].copy()
    print(f"Selected genres: {SELECTED_GENRES}")
    print(f"Rows after genre filtering: {len(df)}")
    return df


def prepare_features_and_labels(df):
    #select the audio features
    X = df[AUDIO_FEATURES].copy()
    y = df["track_genre"].copy()

    #replace missing values
    X = X.fillna(X.median())

    #convert genre names into numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y, encoder


def split_data(X, y):
    #split into train, val and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    #split the rest into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    #fit only on training data to avoid data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler


def prepare_data(filepath="dataset.csv", genres=None, features=None):
    df = load_data(filepath)
    df = filter_genres(df)

    X, y, encoder = prepare_features_and_labels(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

    class_names = encoder.classes_

    print("Data preparation finished")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": AUDIO_FEATURES,
        "class_names": class_names,
        "encoder": encoder,
        "scaler": scaler,
        "n_features": len(AUDIO_FEATURES),
        "n_classes": len(class_names)
    }


if __name__ == "__main__":
    data = prepare_data()
    print(data["X_train"].shape)