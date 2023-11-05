import os
import pickle
import time
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
cd = os.path.dirname(__file__) + '/..'


def load_data(filename):
    df = pd.read_parquet(filename)
    data = df.to_numpy()
    n_samples, n_features = data.shape[0], data.shape[1] - 1
    X, y = data[:, 0:n_features], data[:, n_features]
    return X, y


def evaluate_model(model, X, y, folds):
    y_pred = cross_val_predict(model, X, y, cv=folds)
    roc_auc = round((roc_auc_score(y, cross_val_predict(model, X, y, cv=folds, method='predict_proba')[:, 1]) * 100), 6)
    accuracy = round((accuracy_score(y, y_pred) * 100), 6)
    precision = round((precision_score(y, y_pred) * 100), 6)
    recall = round((recall_score(y, y_pred) * 100), 6)
    f1 = round((f1_score(y, y_pred) * 100), 6)
    con_mat = confusion_matrix(y, y_pred)
    return roc_auc, con_mat, accuracy, precision, recall, f1


def main(n):
    X, y = load_data(f'{cd}/par/pre_processed_dataset.par')
    scores = {'Logistic Regression': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'Random Forest': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'KNN': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'Gaussian Naive Bayes': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'Decision Tree': {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []},
              'Histogram-based Gradient Boosting Classification Tree': {'Accuracy': [], 'Precision': [], 'Recall': [],
                                                                        'F1 Score': [], 'ROC AUC': []}}
    start = time.time()
    for i in range(n):
        print(f'''{i + 1}'th iteration''')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        folds = StratifiedKFold(n_splits=10)

        logreg = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)
        ranfor = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
        navbay = GaussianNB().fit(X_train, y_train)
        dectre = DecisionTreeClassifier().fit(X_train, y_train)
        histclass = HistGradientBoostingClassifier().fit(X_train, y_train)

        models = [
            ('Histogram-based Gradient Boosting Classification Tree', histclass),
            ('Logistic Regression', logreg),
            ('Random Forest', ranfor),
            ('KNN', knn),
            ('Gaussian Naive Bayes', navbay),
            ('Decision Tree', dectre)
        ]

        for model_name, model in models:
            roc_auc, conf_matrix, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, folds)
            fpr, tpr, _ = roc_curve(y_test,
                                    cross_val_predict(model, X_test, y_test, cv=folds, method='predict_proba')[:, 1])
            scores[model_name]['Accuracy'].append(accuracy)
            scores[model_name]['Precision'].append(precision)
            scores[model_name]['Recall'].append(recall)
            scores[model_name]['F1 Score'].append(f1)
            scores[model_name]['ROC AUC'].append(roc_auc)
        i += 1
    end = time.time() - start
    print(f'{end}')
    return scores


if __name__ == "__main__":
    iters = int(input('Enter number of iterations: '))
    with open(f'{cd}/pkl/scores.pkl', 'wb') as handle:
        pickle.dump(main(iters), handle, protocol=pickle.HIGHEST_PROTOCOL)
