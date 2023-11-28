
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Example text data
texts = [
    "Isso é um teste de linguagem",
    "This is a language test",
    "Esto es una prueba de idioma",
    "Dies ist ein Sprachtest"
]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)


# Converte o texto em token, então por exemplo a palavra "is" pode ser o token 1
# Então toda vez que aparecer 1, significa que é a palavra "is"
sequences = tokenizer.texts_to_sequences(texts)
# print("Sequences")
# print(sequences)

# Cria arrays com tamanho máximo de 10, preenche com o número 0 nos espaços vazios
max_sequence_length = 10
train_data = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Example target data (random values for demonstration)
# target_data = np.random.randn(len(texts), 1)
target_data = np.array([1, 2, 3, 4])
print(target_data)
print(len(texts))

# Print shapes of generated data
# print("Padded Sequences Shape:", train_data.shape)
# print("Target Data Shape:", target_data.shape)

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(loss='mse', optimizer='adam')

# Train model
model.fit(train_data, target_data, epochs=4)

# data = [
#     [
#         ["Isso é um teste", "This is a test", "Esto es una prueba", "Das ist ein Test",
#          "Eu gosto de chocolate", "I like chocolate", "Me gusta el chocolate", "Ich mag Schokolade",
#          "Meu amigo é programador", "My friend is a programmer", "Mi amigo es programador", "Mein Freund ist Programmierer",
#          "Ontem eu comi uma pizza de frango", "Yesterday I ate a chicken pizza", "Ayer comí una pizza de pollo", "Gestern habe ich eine Hühnchenpizza gegessen",
#          "O computador é uma máquina incrível", "The computer is an amazing machine", "La computadora es una máquina increíble", "Der Computer ist eine erstaunliche Maschine",
#          "A linguagem é uma forma de se comunicar", "Language is a way to communicate", "El lenguaje es una forma de comunicarse", "Sprache ist eine Möglichkeit zu kommunizieren",
#          "Aula de Processamento de Linguagem Natural é legal", "Natural Language Processing class is cool", "La clase de Procesamiento de Lenguaje Natural es genial", "Der Natural Language Processing-Kurs ist cool"],
#         [1,2,3,4,
#          1,2,3,4,
#          1,2,3,4,
#          1,2,3,4,
#          1,2,3,4,
#          1,2,3,4,
#          1,2,3,4]
#     ]
# ]

data = [
    ["Isso é um teste", 1], ["This is a test",2], ["Esto es una prueba",3], ["Das ist ein Test",4]
]


from sklearn.preprocessing import MinMaxScaler

train_data, test_data = data[:3], data[3:]


sequences = tokenizer.texts_to_sequences(texts)

# Normalize data
# scaler = MinMaxScaler()
# train_data = scaler.fit_transform(train_data)
# test_data = scaler.transform(test_data)




# def create_sequences(data, seq_length):
#     X = []
#     y = []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i+seq_length])
#         y.append(data[i+seq_length])
#     return np.array(X), np.array(y)
#
# # Define sequence length
# seq_length = 12
#
# # Create sequences for training set
# X_train, y_train = create_sequences(train_data, seq_length)
#
# # Create sequences for testing set
# X_test, y_test = create_sequences(test_data, seq_length)
#
# # Reshape input data
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# model = Sequential()
# model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#
#
# model.add(LSTM(units=64, return_sequences=True))
# model.add(LSTM(units=64, return_sequences=True))
#
# model.add(Dense(units=1))
# model.compile(loss='mean_squared_error', optimizer='adam')