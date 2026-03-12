from sklearn.metrics import accuracy_score, f1_score, hamming_loss


def evaluate_models(svm_model, lr_model, X_test, y_test):

    print("Evaluating models...")

    svm_preds = svm_model.predict(X_test)

    lr_probs = lr_model.predict_proba(X_test)

    lr_preds = (lr_probs > 0.5).astype(int)

    svm_metrics = {
        "accuracy": accuracy_score(y_test, svm_preds),
        "f1_micro": f1_score(y_test, svm_preds, average="micro"),
        "f1_macro": f1_score(y_test, svm_preds, average="macro"),
        "hamming_loss": hamming_loss(y_test, svm_preds)
    }

    lr_metrics = {
        "accuracy": accuracy_score(y_test, lr_preds),
        "f1_micro": f1_score(y_test, lr_preds, average="micro"),
        "f1_macro": f1_score(y_test, lr_preds, average="macro"),
        "hamming_loss": hamming_loss(y_test, lr_preds)
    }

    return svm_metrics, lr_metrics