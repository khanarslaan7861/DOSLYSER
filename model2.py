import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def load_data(filename):
    df = pd.read_parquet(filename)
    data = df.to_numpy()
    n_samples, n_features = data.shape[0], data.shape[1] - 1
    X, y = data[:, 0:n_features], data[:, n_features]
    return X, y


def evaluate_model(model, model_name, X, y, folds):
    y_pred = cross_val_predict(model, X, y, cv=folds)
    roc_auc = roc_auc_score(y, cross_val_predict(model, X, y, cv=folds, method='predict_proba')[:, 1]) * 100
    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred) * 100
    recall = recall_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred) * 100
    con_mat = confusion_matrix(y, y_pred)

    print(f'Model: {model_name}')
    print(f"Accuracy: {round(accuracy, 6)}")
    print(f"Precision: {round(precision, 6)}")
    print(f"Recall: {round(recall, 6)}")
    print(f"F1 : {round(f1, 6)}")
    print(f"ROC AUC: {round(roc_auc, 6)}")
    print('===================================================')
    return roc_auc, con_mat


def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f'{model_name}').plot()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')


def plot_confusion_matrix(conf_matrix, classes, model_name):
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(cmap=plt.cm.Reds)
    plt.title(f'Confusion Matrix - {model_name}')


def main():
    X, y = load_data("pre_processed_dataset.par")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    folds = StratifiedKFold(n_splits=10)

    models = [
        ('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)),
        ('Random Forest', RandomForestClassifier(n_estimators=100).fit(X_train, y_train)),
        ('KNN', KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)),
        ('Gaussian Naive Bayes', GaussianNB().fit(X_train, y_train)),
        ('Decision Tree', DecisionTreeClassifier().fit(X_train, y_train))
    ]

    for model_name, model in models:
        roc_auc, conf_matrix = evaluate_model(model, model_name, X_test, y_test, folds)
        fpr, tpr, _ = roc_curve(y_test, cross_val_predict(model, X_test, y_test, cv=folds, method='predict_proba')[:, 1])
        plot_roc_curve(fpr, tpr, roc_auc, model_name)
        plot_confusion_matrix(conf_matrix, model.classes_, model_name)

    plt.show()


if __name__ == "__main__":
    main()
