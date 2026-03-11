from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing import preprocess_genres
from src.feature_engineering import create_features
from src.model_training import train_models
from src.evaluation import evaluate_models


def main():

    print("Starting Genre Prediction Pipeline")

    dataset_path = "data/tv_shows.csv"

    df = load_dataset(dataset_path)

    y, genre_names = preprocess_genres(df)

    X = create_features(df)

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training size:", X_train.shape)
    print("Test size:", X_test.shape)

    svm_model, lr_model = train_models(X_train, y_train)

    svm_metrics, lr_metrics = evaluate_models(
        svm_model,
        lr_model,
        X_test,
        y_test
    )

    print("\nSVM Results")
    print(svm_metrics)

    print("\nLogistic Regression Results")
    print(lr_metrics)


if __name__ == "__main__":
    main()