# ====== Primeiro modelo ========
# Define LSTM model
model1 = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(input_shape, 1)),
    tf.keras.layers.Dense(1)
])

# Compile model
model1.compile(loss='mse', optimizer='adam')
trainScore = model1.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model1.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# Train Score: 0.18 MSE (0.42 RMSE)
# Test Score: 0.17 MSE (0.41 RMSE)

