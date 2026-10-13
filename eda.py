import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import SELECTED_GENRES, AUDIO_FEATURES


def plot_genre_distribution(df):
    genre_counts = df["track_genre"].value_counts()

    plt.figure(figsize=(10, 5))
    genre_counts.plot(kind="bar")

    plt.title("Number of Songs per Genre")
    plt.xlabel("Genre")
    plt.ylabel("Number of Songs")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("results/genre_distribution.png")
    plt.close()


def plot_feature_distributions(df):
    df[AUDIO_FEATURES].hist(figsize=(12, 8), bins=30)

    plt.suptitle("Distribution of Audio Features")
    plt.tight_layout()

    plt.savefig("results/feature_distributions.png")
    plt.close()


def plot_features_by_genre(df):
    important_features = [
        "danceability",
        "energy",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "valence"
    ]

    for feature in important_features:
        plt.figure(figsize=(10, 5))

        sns.boxplot(
            data=df,
            x="track_genre",
            y=feature
        )

        plt.title(f"{feature} by Genre")
        plt.xlabel("Genre")
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(f"results/{feature}_by_genre.png")
        plt.close()


def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))

    correlation = df[AUDIO_FEATURES].corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Correlation Between audio features")
    plt.tight_layout()

    plt.savefig("results/correlation_matrix.png")
    plt.close()


def print_basic_statistics(df):
    print("\nBasic dataset information")
    print(f"Rows: {len(df)}")
    print(f"Genres: {df['track_genre'].nunique()}")
    print(f"Features used: {len(AUDIO_FEATURES)}")

    print("\nGenre distribution:")
    print(df["track_genre"].value_counts())

    print("\nFeature statistics:")
    print(df[AUDIO_FEATURES].describe().round(3))

    print("\nMissing values:")
    print(df[AUDIO_FEATURES].isnull().sum())


def run_eda(filepath="dataset.csv", genres=SELECTED_GENRES):
    df = pd.read_csv(filepath)

    df = df[df["track_genre"].isin(genres)].copy()

    print_basic_statistics(df)

    plot_genre_distribution(df)
    plot_feature_distributions(df)
    plot_features_by_genre(df)
    plot_correlation_matrix(df)


if __name__ == "__main__":
    run_eda()