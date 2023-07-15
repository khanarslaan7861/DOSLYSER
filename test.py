import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score,  \
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("pre_processed_dataset.csv", low_memory=False)
df.drop(columns=df.columns[0], axis=1,  inplace=True)
data = df.to_numpy()
n_samples, n_features = data.shape[0], data.shape[1] - 1
X, y = data[:, 0:n_features], data[:, n_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
folds = StratifiedKFold(n_splits=10)


knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = cross_val_predict(knn, X_test, y_test, cv=folds)
y_score = knn.predict_proba(X_test)[:, 1]
# fpr, tpr, threshold = roc_curve(y_test, y_score)
# roc_auc = roc_auc_score(y_test, y_score)
# print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}")
# print(f"Precision: {precision_score(y_test, y_pred) * 100}")
# print(f"Recall: {recall_score(y_test, y_pred) * 100}")
# print(f"F1 : {f1_score(y_test, y_pred) * 100}")
# print(f"ROC AUC: {round((roc_auc * 100), 6)}")
RocCurveDisplay.from_estimator(knn, X_test, y_test)

#
# con_mat = confusion_matrix(y_test, y_pred)
# (ConfusionMatrixDisplay(con_mat, display_labels=knn.classes_)).plot(cmap=plt.cm.Reds)
plt.show()

models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', multi_class='ovr'),
    "Random Forest Classifier with 100 estimators": RandomForestClassifier(n_estimators=100),
    "K Nearest Neighbour Classifier": KNeighborsClassifier(n_neighbors=1),
    "Gaussian Naive Bayes Classifier": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

for i in models:
    # model = models[i]
    # print(model)
    pass
