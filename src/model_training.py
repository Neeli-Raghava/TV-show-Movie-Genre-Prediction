from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_models(X_train, y_train):

    print("Training SVM model...")

    svm = OneVsRestClassifier(
        LinearSVC(class_weight="balanced")
    )

    param_grid_svm = {
        "estimator__C": [0.1, 1, 3],
        "estimator__max_iter": [2000, 3000]
    }

    grid_svm = GridSearchCV(
        svm,
        param_grid_svm,
        scoring="f1_micro",
        cv=5,
        n_jobs=-1
    )

    grid_svm.fit(X_train, y_train)

    print("Best SVM params:", grid_svm.best_params_)

    print("Training Logistic Regression model...")

    lr = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )

    param_grid_lr = {
        "estimator__C": [0.1, 1, 5],
        "estimator__solver": ["lbfgs"]
    }

    grid_lr = GridSearchCV(
        lr,
        param_grid_lr,
        scoring="f1_micro",
        cv=5,
        n_jobs=-1
    )

    grid_lr.fit(X_train, y_train)

    print("Best LR params:", grid_lr.best_params_)

    return grid_svm.best_estimator_, grid_lr.best_estimator_
