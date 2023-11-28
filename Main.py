import math

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Example text data
data = [
    ["Oi, tudo bem?", 1], ["Hey how's it going?", 2], ["¿Hola todo bien?", 3], ["Hey, wie geht's?", 4],
    ["Me chamo Gabriel", 1], ["My name is Gabriel", 2], ["Mi nombre es Gabriel", 3], ["Ich heiße Gabriel", 4],
    ["Vou para a praia", 1], ["I'm going to the beach", 2], ["Voy a la playa", 3], ["Ich gehe zum Strand", 4],
    ["Andar de carro", 1], ["Ride in a car", 2], ["Montar en un coche", 3], ["In einem Auto fahren", 4],
    ["Visitar meus familiares", 1], ["Visit my family", 2], ["Visitar mi familia", 3], ["Besuche meine Familie", 4],
    ["Estender a roupa", 1], ["Hang out clothes", 2], ["Tender la ropa", 3], ["Kleidung aufhängen", 4],
    ["Comer pizza", 1], ["Eat pizza", 2], ["Comer pizza", 3], ["Pizza essen", 4],
    ["Aprender um novo idioma", 1], ["Learn a new language", 2], ["Aprender un nuevo idioma", 3],
    ["Eine neue Sprache lernen", 4],
    ["Assistir a um filme", 1], ["Watch a movie", 2], ["Ver una película", 3], ["Einen Film anschauen", 4],
    ["Praticar esportes", 1], ["Play sports", 2], ["Practicar deportes", 3], ["Sport treiben", 4],
    ["Ler um livro", 1], ["Read a book", 2], ["Leer un libro", 3], ["Ein Buch lesen", 4],
    ["Cozinhar uma nova receita", 1], ["Cook a new recipe", 2], ["Cocinar una nueva receta", 3],
    ["Ein neues Rezept kochen", 4],
    ["Ouvir música", 1], ["Listen to music", 2], ["Escuchar música", 3], ["Musik hören", 4],
    ["Viajar para um lugar exótico", 1], ["Travel to an exotic place", 2], ["Viajar a un lugar exótico", 3],
    ["Zu einem exotischen Ort reisen", 4],
    ["Praticar yoga", 1], ["Practice yoga", 2], ["Practicar yoga", 3], ["Yoga praktizieren", 4],
    ["Trabalhar em casa", 1], ["Work from home", 2], ["Trabajar desde casa", 3], ["Von zu Hause aus arbeiten", 4],
    ["Conversar com amigos", 1], ["Talk to friends", 2], ["Conversar con amigos", 3], ["Mit Freunden sprechen", 4],
    ["Fazer uma caminhada", 1], ["Take a walk", 2], ["Dar un paseo", 3], ["Einen Spaziergang machen", 4],
    ["Estudar para um exame", 1], ["Study for an exam", 2], ["Estudiar para un examen", 3],
    ["Für eine Prüfung lernen", 4],
    ["Planejar as férias", 1], ["Plan the holidays", 2], ["Planear las vacaciones", 3], ["Die Ferien planen", 4],
    ["Cantar uma música", 1], ["Sing a song", 2], ["Cantar una canción", 3], ["Ein Lied singen", 4],
    ["Fazer compras", 1], ["Go shopping", 2], ["Ir de compras", 3], ["Einkaufen gehen", 4],
    ["Aprender a tocar um instrumento", 1], ["Learn to play an instrument", 2], ["Aprender a tocar un instrumento", 3],
    ["Lernen, ein Instrument zu spielen", 4],
    ["Assistir a um jogo de futebol", 1], ["Watch a soccer game", 2], ["Ver un partido de fútbol", 3],
    ["Ein Fußballspiel anschauen", 4],
    ["Escrever um poema", 1], ["Write a poem", 2], ["Escribir un poema", 3], ["Ein Gedicht schreiben", 4],
    ["Experimentar uma nova receita", 1], ["Try a new recipe", 2], ["Probar una nueva receta", 3],
    ["Ein neues Rezept ausprobieren", 4],
    ["Aprender a dançar", 1], ["Learn to dance", 2], ["Aprender a bailar", 3], ["Tanzen lernen", 4],
    ["Assistir a um pôr do sol", 1], ["Watch a sunset", 2], ["Ver un atardecer", 3],
    ["Einen Sonnenuntergang anschauen", 4],
    ["Correr no parque", 1], ["Run in the park", 2], ["Correr en el parque", 3], ["Im Park laufen", 4],
    ["Fazer uma videochamada", 1], ["Make a video call", 2], ["Hacer una videollamada", 3],
    ["Einen Videoanruf tätigen", 4],
    ["Visitar um museu", 1], ["Visit a museum", 2], ["Visitar un museo", 3], ["Ein Museum besuchen", 4],
    ["Aprender a programar", 1], ["Learn to code", 2], ["Aprender a programar", 3], ["Programmieren lernen", 4],
    ["Ir ao cinema", 1], ["Go to the movies", 2], ["Ir al cine", 3], ["Ins Kino gehen", 4],
    ["Pintar um quadro", 1], ["Paint a picture", 2], ["Pintar un cuadro", 3], ["Ein Bild malen", 4],
    ["Praticar meditação", 1], ["Practice meditation", 2], ["Practicar meditación", 3], ["Meditation üben", 4]
]

# Tokenize the text data
tokenizer = Tokenizer()
texts = np.array([item[0] for item in data])
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
target_data = np.array([item[1] for item in data])
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

teste = [["Fiz um exercício físico hoje", 1], ["Run in the park", 2], ["Correr en el parque", 3], ["Im Park laufen", 4]]
test_X = np.array([item[0] for item in teste])
test_Y = np.array([item[1] for item in teste])
tokenizer.fit_on_texts(test_X)
test_X = tokenizer.texts_to_sequences(test_X)
test_X = pad_sequences(test_X, maxlen=max_sequence_length, padding='post', truncating='post')

testScore = model.evaluate(test_X, test_Y, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))