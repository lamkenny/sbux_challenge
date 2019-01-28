import pickle
import bz2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer

def calculate_dcg(y, k):
    """Calculate DCG at k"""
    rel = np.take(y, np.arange(k))
    gains = 2 ** rel - 1
    # i starts at 1 while range/index starts at 0, so plus 1
    discounts = np.log2((np.arange(len(rel)) + 1) + 1)
    return np.sum(gains/discounts)

def calculate_ndcg(y_true, y_pred, k):
    """Calculate nDCG at k"""
    ranked_indices = np.argsort(y_pred)[::-1]
    rel = np.take(y_true, ranked_indices)
    dcg = calculate_dcg(rel, k)

    ranked_indices = np.argsort(y_true)[::-1]
    rel = np.take(y_true, ranked_indices)
    idcg = calculate_dcg(rel, k)
    return dcg/idcg

def evaluate(y_true, y_pred):
    """Evaluate y_true against y_pred using ndcg@3"""
    k=3
    lb = LabelBinarizer()
    lb.fit(range(y_pred.shape[1] + 1))
    binarized_labels = lb.transform(y_true)

    scores = []
    for y_true, y_pred in zip(binarized_labels, y_pred):
        scores.append(calculate_ndcg(y_true, y_pred, k))
        
    return np.mean(scores)


X_train = pd.read_csv('data/clean/X_train.csv', index_col=0)
y_train = pd.read_csv('data/clean/y_train.csv', index_col=0)
X_test = pd.read_csv('data/clean/X_test.csv', index_col=0)
y_test = pd.read_csv('data/clean/y_test.csv', index_col=0)

rf = RandomForestClassifier(bootstrap=True, max_depth=25, max_features='auto', min_samples_leaf=2, min_samples_split=10, n_estimators=2000, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

print('\n')
print('nDCG@3: {}'.format(evaluate(y_test, rf.predict_proba(X_test))))
print('\n')

# bzip the model because a dump can be too large for some filesystems, such as one on Mac OSX that it causes an error.
model_filename = 'models/rf.pkl.bz2'
print('Saving model to {}'.format(model_filename))
with bz2.BZ2File(model_filename, 'w') as file:
    pickle.dump(rf, file)
print('Saving model to {} completed'.format(model_filename))
