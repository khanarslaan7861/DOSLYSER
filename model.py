import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt

# Data Feature Split
df = pd.read_csv("pre_processed_dataset.csv", low_memory=False)
data = df.to_numpy()
n_samples, n_features = data.shape[0], data.shape[1] - 1
x, y = data[:, 0:n_features], data[:, n_features]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3)

# Stratified K Fold cross validation
folds = StratifiedKFold(n_splits=10)


# Model test function

def test(model, x_tr, x_te, y_tr, y_te, fold):
    y_pred = cross_val_predict(model, x_tr, y_tr, cv=fold)
    print(f"Accuracy: {accuracy_score(y_pred, y_tr) * 100}")
    print(f"Precision: {precision_score(y_tr, y_pred) * 100}")
    print(f"Recall: {recall_score(y_tr, y_pred) * 100}")
    print(f"F1 : {f1_score(y_tr, y_pred) * 100}")
    con_matrix = confusion_matrix(y_tr, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=con_matrix, figsize=(6, 6), cmap=plt.cm.Reds)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


print('Logistic Regression')
test(LogisticRegression(solver='liblinear', multi_class='ovr'), x_train, x_test, y_train, y_test, folds)
print()

print('Random Forest Classifier with 10 estimators')
test(RandomForestClassifier(n_estimators=10), x_train, x_test, y_train, y_test, folds)
print()

print('Random Forest Classifier with 100 estimators')
test(RandomForestClassifier(n_estimators=100), x_train, x_test, y_train, y_test, folds)
print()

print('K Nearest Neighbour Classifier')
test(KNeighborsClassifier(n_neighbors=1), x_train, x_test, y_train, y_test, folds)
print()

print('Gaussian Naive Bayes Classifier')
test(GaussianNB(), x_train, x_test, y_train, y_test, folds)
print()

print('Decision Tree')
test(DecisionTreeClassifier(), x_train, x_test, y_train, y_test, folds)
