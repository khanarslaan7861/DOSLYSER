import pickle
import os
cd = os.path.dirname(__file__) + '/..'
pkl = os.listdir(f'{cd}/pkl/')

models = [
        ('Logistic Regression', logreg),
        ('Random Forest', ranfor),
        ('KNN', knn),
        ('Gaussian Naive Bayes', navbay),
        ('Decision Tree', dectre)
    ]

