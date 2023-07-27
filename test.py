import pickle

import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
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
    roc_auc = round((roc_auc_score(y, cross_val_predict(model, X, y, cv=folds, method='predict_proba')[:, 1]) * 100), 6)
    accuracy = round((accuracy_score(y, y_pred) * 100), 6)
    precision = round((precision_score(y, y_pred) * 100), 6)
    recall = round((recall_score(y, y_pred) * 100), 6)
    f1 = round((f1_score(y, y_pred) * 100), 6)
    con_mat = confusion_matrix(y, y_pred)

    # print(f'Model: {model_name}')
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 : {f1}")
    # print(f"ROC AUC: {roc_auc}")
    # print('===================================================')
    return roc_auc, con_mat, accuracy, precision, recall, f1


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
    scores = {'Logistic Regression': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'Random Forest': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'KNN': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'Gaussian Naive Bayes': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'Decision Tree': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []}}

    i = 0
    while i <= 2:
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
            roc_auc, conf_matrix, accuracy, precision, recall, f1 = evaluate_model(model, model_name, X_test, y_test,
                                                                                   folds)
            fpr, tpr, _ = roc_curve(y_test,
                                    cross_val_predict(model, X_test, y_test, cv=folds, method='predict_proba')[:, 1])
            # plot_roc_curve(fpr, tpr, roc_auc, model_name)
            # plot_confusion_matrix(conf_matrix, model.classes_, model_name)
            scores[model_name]['Accuracy'].append(accuracy)
            scores[model_name]['Precision'].append(precision)
            scores[model_name]['Recall'].append(recall)
            scores[model_name]['F1 Score'].append(f1)
            scores[model_name]['ROC AUC'].append(roc_auc)
        i += 1
    return scores


# plt.show()

if __name__ == "__main__":
    with open('scores.pkl', 'wb') as handle:
        pickle.dump(main(), handle, protocol=pickle.HIGHEST_PROTOCOL)
