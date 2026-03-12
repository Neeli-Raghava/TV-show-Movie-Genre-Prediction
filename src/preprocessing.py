from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_genres(df):

    print("Processing genre labels...")

    mlb = MultiLabelBinarizer()

    genre_list = df["listed_in"].apply(
        lambda x: [i.strip() for i in str(x).split(",")]
    )

    y = mlb.fit_transform(genre_list)

    genre_names = mlb.classes_

    genre_counts = y.sum(axis=0)

    min_samples = 50

    valid_genres = genre_counts >= min_samples

    y = y[:, valid_genres]

    genre_names = genre_names[valid_genres]

    print("Genres before filtering:", len(genre_counts))
    print("Genres after filtering:", len(genre_names))

    return y, genre_names