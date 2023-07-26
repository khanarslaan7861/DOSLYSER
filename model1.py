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

df = pd.read_csv("pre_processed_dataset.par")
data = df.to_numpy()
n_samples, n_features = data.shape[0], data.shape[1] - 1
X, y = data[:, 0:n_features], data[:, n_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
folds = StratifiedKFold(n_splits=10)


def scoring(model):
    y_pred = cross_val_predict(model, X_test, y_test, cv=folds)
    y_score = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_score) * 100
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_cur = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f'{model}')
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    con_mat = confusion_matrix(y_test, y_pred)
    print('===================================================')
    print(f'Model: {model}')
    print(f"Accuracy: {round(accuracy, 6)}")
    print(f"Precision: {round(precision, 6)}")
    print(f"Recall: {round(recall, 6)}")
    print(f"F1 : {round(f1, 6)}")
    print(f"ROC AUC: {round(roc_auc, 6)}")
    roc_cur.plot()
    ConfusionMatrixDisplay(con_mat, display_labels=model.classes_).plot(cmap=plt.cm.Reds)
    plt.ion()
    plt.show()
    plt.pause(0.01)


logistic_regression = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)
random_forest = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
gaussian_naive_bayes = GaussianNB().fit(X_train, y_train)
decision_tree = DecisionTreeClassifier().fit(X_train, y_train)

scoring(logistic_regression)
scoring(random_forest)
scoring(knn)
scoring(gaussian_naive_bayes)
scoring(decision_tree)
plt.pause(1000)
