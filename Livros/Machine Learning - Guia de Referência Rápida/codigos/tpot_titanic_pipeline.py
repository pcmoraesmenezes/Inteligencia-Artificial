from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('/home/pcmoraes/Área de Trabalho/Editores de código/Inteligencia-Artificial/Livros/Machine Learning - Guia de Referência Rápida/datasets/titanic.csv')
features = tpot_data.drop('target', axis=1)  # Change 'survived' to 'target'
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)
# Average CV score on the training set was: 0.8133523402233308
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.45, min_samples_leaf=5, min_samples_split=4, n_estimators=100)),
    SelectPercentile(score_func=f_classif, percentile=42),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

accuracy = accuracy_score(testing_target, results)
print(f'Accuracy: {accuracy:.2f}')