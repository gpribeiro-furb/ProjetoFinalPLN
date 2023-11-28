import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def mainMethod(data):
    # Selecionando os dados do dataset
    texts = np.array([item[0] for item in data])
    results = np.array([item[1] for item in data])

    # Criando o objeto para tokenização
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # Converte o texto em token, então por exemplo a palavra "is" pode ser o token 1
    # Então toda vez que aparecer 1, significa que é a palavra "is"
    sequences = tokenizer.texts_to_sequences(texts)

    # Cria arrays com tamanho máximo de 20, preenche com o número 0 nos espaços vazios
    input_shape = 20
    sequences = pad_sequences(sequences, maxlen=input_shape, padding='post', truncating='post')

    # Embaralhando, e separando o dataset em 80% de treino e 20% de teste
    separacao = int(len(sequences)*0.8)
    np.random.shuffle(sequences)
    np.random.shuffle(results)
    X_train, X_test = sequences[:separacao], sequences[separacao:]
    y_train, y_test = results[:separacao], results[separacao:]

    # Criando o modelo com várias camadas
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Treinando
    model.fit(X_train, y_train, epochs=10, batch_size=1)

    # Resultados
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Arquivos
path_2Linguas = 'datasets\dataset2Linguas.json'
path_4Linguas = 'datasets\dataset4Linguas.json'
path_binario = 'datasets\datasetBinario.json'

with open(path_4Linguas, 'r') as file:
    data = json.load(file)
    mainMethod(data)

with open(path_2Linguas, 'r') as file:
    data = json.load(file)
    mainMethod(data)

with open(path_binario, 'r') as file:
    data = json.load(file)
    mainMethod(data)