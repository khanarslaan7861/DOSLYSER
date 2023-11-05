import pickle
import os

cd = os.path.dirname(__file__) + '/../pkl/scores.pkl'

with open(cd, 'rb') as file:
    pkl = pickle.load(file)
    print(pkl)
