try:
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import accuracy_score, classification_report  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
except Exception as e:  # pragma: no cover - fallback for import issues
    raise ImportError(
        "Could not import required modules from scikit-learn. "
        "Ensure scikit-learn is installed and up to date.") from e


def train_supervised_model(X, y):
    """
    Entraîne un modèle supervisé si un label est disponible.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importances": dict(zip(X.columns, model.feature_importances_))
    }

    return results