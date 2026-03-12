import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import pandas as pd


def extract_duration(val):

    try:

        if "min" in str(val):
            return int(str(val).replace(" min", ""))

        elif "Season" in str(val):
            return int(str(val).split(" ")[0]) * 400

        return 0

    except:
        return 0


def create_features(df):

    print("Creating TF-IDF features...")

    tfidf_desc = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5
    )

    tfidf_title = TfidfVectorizer(
        max_features=1000,
        stop_words="english"
    )

    X_desc = tfidf_desc.fit_transform(df["description"])

    X_title = tfidf_title.fit_transform(df["title"].fillna(""))

    print("Extracting metadata features...")

    df["Duration_Mins"] = df["duration"].apply(extract_duration)

    df["rating"] = df["rating"].fillna("Unknown")

    type_dummies = pd.get_dummies(df["type"], prefix="type")

    rating_dummies = pd.get_dummies(df["rating"], prefix="rate")

    if "platform" in df.columns:
        platform_dummies = pd.get_dummies(df["platform"], prefix="plat")
    else:
        platform_dummies = pd.DataFrame()

    X_num = df[["release_year", "Duration_Mins"]].copy()

    X_num["release_year"] = X_num["release_year"].fillna(
        X_num["release_year"].median()
    )

    scaler = StandardScaler()

    X_num_scaled = scaler.fit_transform(X_num)

    X_metadata = np.hstack([
        X_num_scaled,
        type_dummies.values,
        rating_dummies.values,
        platform_dummies.values if not platform_dummies.empty else np.empty((len(df), 0))
    ])

    X_final = hstack([
        X_desc,
        X_title,
        X_metadata
    ])

    print("Final feature shape:", X_final.shape)

    return X_final