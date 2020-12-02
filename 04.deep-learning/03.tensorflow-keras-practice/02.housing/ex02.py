
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

df = pd.read_csv("./dataset/housing.csv", delim_whitespace=True, header=None)
dataset = df.values
x = dataset[:, 0:13]
t = dataset[:, 13]

train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=3)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_x, train_t, epochs=200, batch_size=10)

# 예측 값과 실제 값의 비교
y = model.predict(test_x).flatten()
for i in range(10):
    label = test_t[i]
    prediction = y[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
