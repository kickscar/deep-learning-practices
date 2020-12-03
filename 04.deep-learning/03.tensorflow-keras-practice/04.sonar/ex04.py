# coding: utf-8
# Sonar Mineral Binary Classification Model
# Model Fitting #3 - K-Folder Cross Validation
from keras.models import Sequential

from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# 1. load training/test data
from tensorflow.python.keras.utils.np_utils import to_categorical

df = pd.read_csv('./dataset/sonar.csv', header=None)

dataset = df.values
x = dataset[:, 0:60].astype(float)

t = dataset[:, 60]
e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = to_categorical(t)

# 2. 10-fold cross validation
nfold = 10
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0)
accuracies = []

for train, test in skf.split(x, t):
    # 2. model frame config
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    # 3. model fitting environment
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 4. model fitting
    model.fit(x[train], t[train], epochs=100, batch_size=5, verbose=1)

    # 5. evaluate
    result = model.evaluate(x[test], t[test], verbose=0)
    accuracies.append(result[1])


# result
print(f'\n {nfold} fold accuracy: {accuracies}')
