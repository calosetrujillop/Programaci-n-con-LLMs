def evaluar_con_confusion_acumulada(X, y, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    n_classes = len(np.unique(y))
    confusion_total = np.zeros((n_classes, n_classes), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)

        accuracies.append(accuracy_score(y_test, y_pred))
        confusion_total += confusion_matrix(
            y_test, y_pred, labels=np.arange(n_classes)
        )

    return {
        'accuracy_promedio': round(float(np.mean(accuracies)), 4),
        'confusion_acumulada': confusion_total
    }
