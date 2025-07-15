import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

ld = pd.read_csv("Lowered")
lowerData = ld.to_numpy()
rd = pd.read_csv("Raised")
raisedData = rd.to_numpy()

y1 = np.zeros(lowerData.shape[0])
y2 = np.ones(raisedData.shape[0])

Y = np.append(y1, y2, axis=0)
X = np.append(lowerData, raisedData, axis=0)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
Y = Y[indices]

XDev = X[0:100]
YDev = Y[0:100]
XTrain = X[100:]
YTrain = Y[100:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=X.shape[1]),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation ='sigmoid')  # 2 output classes
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
history = model.fit(XTrain, YTrain, epochs=50, batch_size=64, validation_data=(XDev, YDev), callbacks=[callback])
model.evaluate(XDev, YDev, verbose=2)

# Predict
probability_model = tf.keras.Sequential([
    model
])
predictions = probability_model(XDev[:5])
print(predictions)

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Dev Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()