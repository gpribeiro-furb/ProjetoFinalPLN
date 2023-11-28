import math

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Example text data
data = [
    ["Oi, tudo bem?", 10],
    ["Hey how's it going?", 20],
    ["¿Hola todo bien?", 30],
    ["Hey, wie geht's?", 40],
    ["Me chamo Gabriel", 10],
    ["My name is Gabriel", 20],
    ["Mi nombre es Gabriel", 30],
    ["Ich heiße Gabriel", 40],
    ["Vou para a praia", 10],
    ["I'm going to the beach", 20],
    ["Voy a la playa", 30],
    ["Ich gehe zum Strand", 40],
    ["Andar de carro", 10],
    ["Ride in a car", 20],
    ["Montar en un coche", 30],
    ["In einem Auto fahren", 40],
    ["Visitar meus familiares", 10],
    ["Visit my family", 20],
    ["Visitar mi familia", 30],
    ["Besuche meine Familie", 40],
    ["Estender a roupa", 10],
    ["Hang out clothes", 20],
    ["Tender la ropa", 30],
    ["Kleidung aufhängen", 40],
    ["Comer pizza", 10],
    ["Eat pizza", 20],
    ["Comer pizza", 30],
    ["Pizza essen", 40],
    ["Aprender um novo idioma", 10],
    ["Learn a new language", 20],
    ["Aprender un nuevo idioma", 30],
    ["Eine neue Sprache lernen", 40],
    ["Assistir a um filme", 10],
    ["Watch a movie", 20],
    ["Ver una película", 30],
    ["Einen Film anschauen", 40],
    ["Praticar esportes", 10],
    ["Play sports", 20],
    ["Practicar deportes", 30],
    ["Sport treiben", 40],
    ["Ler um livro", 10],
    ["Read a book", 20],
    ["Leer un libro", 30],
    ["Ein Buch lesen", 40],
    ["Cozinhar uma nova receita", 10],
    ["Cook a new recipe", 20],
    ["Cocinar una nueva receta", 30],
    ["Ein neues Rezept kochen", 40],
    ["Ouvir música", 10],
    ["Listen to music", 20],
    ["Escuchar música", 30],
    ["Musik hören", 40],
    ["Viajar para um lugar exótico", 10],
    ["Travel to an exotic place", 20],
    ["Viajar a un lugar exótico", 30],
    ["Zu einem exotischen Ort reisen", 40],
    ["Praticar yoga", 10],
    ["Practice yoga", 20],
    ["Practicar yoga", 30],
    ["Yoga praktizieren", 40],
    ["Trabalhar em casa", 10],
    ["Work from home", 20],
    ["Trabajar desde casa", 30],
    ["Von zu Hause aus arbeiten", 40],
    ["Conversar com amigos", 10],
    ["Talk to friends", 20],
    ["Conversar con amigos", 30],
    ["Mit Freunden sprechen", 40],
    ["Fazer uma caminhada", 10],
    ["Take a walk", 20],
    ["Dar un paseo", 30],
    ["Einen Spaziergang machen", 40],
    ["Estudar para um exame", 10],
    ["Study for an exam", 20],
    ["Estudiar para un examen", 30],
    ["Für eine Prüfung lernen", 40],
    ["Planejar as férias", 10],
    ["Plan the holidays", 20],
    ["Planear las vacaciones", 30],
    ["Die Ferien planen", 40],
    ["Cantar uma música", 10],
    ["Sing a song", 20],
    ["Cantar una canción", 30],
    ["Ein Lied singen", 40],
    ["Fazer compras", 10],
    ["Go shopping", 20],
    ["Ir de compras", 30],
    ["Einkaufen gehen", 40],
    ["Aprender a tocar um instrumento", 10],
    ["Learn to play an instrument", 20],
    ["Aprender a tocar un instrumento", 30],
    ["Lernen, ein Instrument zu spielen", 40],
    ["Assistir a um jogo de futebol", 10],
    ["Watch a soccer game", 20],
    ["Ver un partido de fútbol", 30],
    ["Ein Fußballspiel anschauen", 40],
    ["Escrever um poema", 10],
    ["Write a poem", 20],
    ["Escribir un poema", 30],
    ["Ein Gedicht schreiben", 40],
    ["Experimentar uma nova receita", 10],
    ["Try a new recipe", 20],
    ["Probar una nueva receta", 30],
    ["Ein neues Rezept ausprobieren", 40],
    ["Aprender a dançar", 10],
    ["Learn to dance", 20],
    ["Aprender a bailar", 30],
    ["Tanzen lernen", 40],
    ["Assistir a um pôr do sol", 10],
    ["Watch a sunset", 20],
    ["Ver un atardecer", 30],
    ["Einen Sonnenuntergang anschauen", 40],
    ["Correr no parque", 10],
    ["Run in the park", 20],
    ["Correr en el parque", 30],
    ["Im Park laufen", 40],
    ["Fazer uma videochamada", 10],
    ["Make a video call", 20],
    ["Hacer una videollamada", 30],
    ["Einen Videoanruf tätigen", 40],
    ["Visitar um museu", 10],
    ["Visit a museum", 20],
    ["Visitar un museo", 30],
    ["Ein Museum besuchen", 40],
    ["Aprender a programar", 10],
    ["Learn to code", 20],
    ["Aprender a programar", 30],
    ["Programmieren lernen", 40],
    ["Ir ao cinema", 10],
    ["Go to the movies", 20],
    ["Ir al cine", 30],
    ["Ins Kino gehen", 40],
    ["Pintar um quadro", 10],
    ["Paint a picture", 20],
    ["Pintar un cuadro", 30],
    ["Ein Bild malen", 40],
    ["Praticar meditação", 10],
    ["Practice meditation", 20],
    ["Practicar meditación", 30],
    ["Meditation üben", 40],
    ["Explorar novos lugares permite descobrir culturas fascinantes e criar memórias duradouras", 10],
    ["Exploring new places allows you to discover fascinating cultures and create lasting memories", 20],
    ["Explorar nuevos lugares te permite descubrir culturas fascinantes y crear recuerdos duraderos", 30],
    [
        "Das Erkunden neuer Orte ermöglicht es, faszinierende Kulturen zu entdecken und bleibende Erinnerungen zu schaffen",
        40],
    ["Aprender a tocar um instrumento musical requer paciência, prática constante e dedicação apaixonada", 10],
    ["Learning to play a musical instrument requires patience, constant practice, and passionate dedication", 20],
    ["Aprender a tocar un instrumento musical requiere paciencia, práctica constante y dedicación apasionada", 30],
    ["Das Erlernen eines Musikinstruments erfordert Geduld, ständiges Üben und leidenschaftliche Hingabe", 40],
    ["Apreciar a natureza nos lembra da beleza simples que muitas vezes passamos despercebidos na vida cotidiana", 10],
    ["Appreciating nature reminds us of the simple beauty that often goes unnoticed in everyday life", 20],
    ["Apreciar la naturaleza nos recuerda la belleza simple que a menudo pasa desapercibida en la vida cotidiana", 30],
    ["Die Natur zu schätzen erinnert uns an die einfache Schönheit, die im Alltag oft übersehen wird", 40],
    ["Escrever em um diário é uma maneira terapêutica de expressar pensamentos, sentimentos e reflexões pessoais", 10],
    ["Writing in a journal is a therapeutic way to express thoughts, feelings, and personal reflections", 20],
    ["Escribir en un diario es una forma terapéutica de expresar pensamientos, sentimientos y reflexiones personales",
     30],
    [
        "Das Schreiben in einem Tagebuch ist eine therapeutische Möglichkeit, Gedanken, Gefühle und persönliche Reflexionen auszudrücken",
        40],
    ["A arte de cozinhar vai além de simplesmente preparar alimentos, é uma forma de expressão criativa e cultural",
     10],
    ["The art of cooking goes beyond simply preparing food; it is a form of creative and cultural expression", 20],
    ["El arte de cocinar va más allá de simplemente preparar alimentos; es una forma de expresión creativa y cultural",
     30],
    [
        "Die Kunst des Kochens geht über das einfache Zubereiten von Lebensmitteln hinaus; es ist eine Form kreativen und kulturellen Ausdrucks",
        40],
    [
        "A prática regular de meditação traz benefícios para a saúde mental, reduzindo o estresse e promovendo a serenidade interior",
        10],
    ["Regular meditation practice brings mental health benefits, reducing stress and promoting inner serenity", 20],
    [
        "La práctica regular de la meditación aporta beneficios para la salud mental, reduce el estrés y promueve la serenidad interior",
        30],
    [
        "Die regelmäßige Praxis der Meditation bringt Vorteile für die psychische Gesundheit, reduziert Stress und fördert inneren Frieden",
        40],
    ["Acompanhar o ritmo da música durante a dança é uma forma de se conectar com a energia do momento", 10],
    ["Keeping up with the rhythm of the music while dancing is a way to connect with the energy of the moment", 20],
    ["Seguir el ritmo de la música mientras bailas es una forma de conectarse con la energía del momento", 30],
    ["Im Takt der Musik zu tanzen ist eine Möglichkeit, sich mit der Energie des Moments zu verbinden", 40],
    ["A troca de experiências culturais enriquece a compreensão global e promove a tolerância e a aceitação mútua", 10],
    ["The exchange of cultural experiences enriches global understanding and promotes mutual tolerance and acceptance",
     20],
    [
        "El intercambio de experiencias culturales enriquece la comprensión global y promueve la tolerancia y la aceptación mutua",
        30],
    [
        "Der Austausch kultureller Erfahrungen bereichert das globale Verständnis und fördert gegenseitige Toleranz und Akzeptanz",
        40],
    ["Amar a si mesmo é o primeiro passo para cultivar relacionamentos saudáveis e uma vida plena", 10],
    ["Loving oneself is the first step to cultivating healthy relationships and a fulfilling life", 20],
    ["Amarse a uno mismo es el primer paso para cultivar relaciones saludables y una vida plena", 30],
    ["Sich selbst zu lieben ist der erste Schritt zur Pflege gesunder Beziehungen und eines erfüllten Lebens", 40],
    ["A prática regular de exercícios físicos não apenas fortalece o corpo, mas também melhora o humor e a disposição",
     10],
    ["Regular exercise not only strengthens the body but also improves mood and energy levels", 20],
    [
        "El ejercicio regular no solo fortalece el cuerpo, sino que también mejora el estado de ánimo y los niveles de energía",
        30],
    [
        "Regelmäßige körperliche Betätigung stärkt nicht nur den Körper, sondern verbessert auch die Stimmung und die Energielevels",
        40],
    [
        "A contemplação de obras de arte inspira a imaginação, estimula a criatividade e proporciona uma experiência estética única",
        10],
    [
        "Contemplating works of art inspires imagination, stimulates creativity, and provides a unique aesthetic experience",
        20],
    [
        "La contemplación de obras de arte inspira la imaginación, estimula la creatividad y proporciona una experiencia estética única",
        30],
    [
        "Das Betrachten von Kunstwerken inspiriert die Vorstellungskraft, regt die Kreativität an und bietet eine einzigartige ästhetische Erfahrung",
        40],
    ["A dedicação ao aprendizado contínuo é o caminho para o crescimento pessoal e o desenvolvimento profissional", 10],
    ["Dedication to continuous learning is the path to personal growth and professional development", 20],
    ["La dedicación al aprendizaje continuo es el camino hacia el crecimiento personal y el desarrollo profesional",
     30],
    ["Die Hingabe zum kontinuierlichen Lernen ist der Weg zum persönlichen Wachstum und beruflichen Fortschritt", 40],
    [
        "Explorar a diversidade culinária amplia nossos horizontes gastronômicos e nos conecta a diferentes tradições",
        10],
    ["Exploring culinary diversity broadens our gastronomic horizons and connects us to different traditions", 20],
    [
        "Explorar la diversidad culinaria amplía nuestros horizontes gastronómicos y nos conecta con diferentes tradiciones",
        30],
    [
        "Die Erkundung kulinarischer Vielfalt erweitert unsere gastronomischen Horizonte und verbindet uns mit verschiedenen Traditionen",
        40],
    ["Cultivar amizades genuínas requer tempo, empatia e um coração aberto para as experiências do outro", 10],
    ["Cultivating genuine friendships takes time, empathy, and an open heart to each other's experiences", 20],
    ["Cultivar amistades genuinas requiere tiempo, empatía y un corazón abierto a las experiencias del otro", 30],
    ["Echte Freundschaften pflegen erfordert Zeit, Empathie und ein offenes Herz für die Erfahrungen des anderen", 40],
    ["Adotar uma mentalidade positiva é a chave para superar desafios e encontrar soluções construtivas", 10],
    ["Embracing a positive mindset is the key to overcoming challenges and finding constructive solutions", 20],
    ["Adoptar una mentalidad positiva es la clave para superar desafíos y encontrar soluciones constructivas", 30],
    [
        "Eine positive Denkweise annehmen ist der Schlüssel, um Herausforderungen zu überwinden und konstruktive Lösungen zu finden",
        40],
    ["Aprender um novo idioma amplia nossas capacidades de comunicação e enriquece nossa perspectiva global", 10],
    ["Learning a new language expands our communication skills and enriches our global perspective", 20],
    ["Aprender un nuevo idioma amplía nuestras habilidades de comunicación y enriquece nuestra perspectiva global", 30],
    [
        "Das Erlernen einer neuen Sprache erweitert unsere Kommunikationsfähigkeiten und bereichert unsere globale Perspektive",
        40],
    ["Valorizar momentos de tranquilidade contribui para o equilíbrio emocional e a paz interior", 10],
    ["Appreciating moments of tranquility contributes to emotional balance and inner peace", 20],
    ["Valorar momentos de tranquilidad contribuye al equilibrio emocional y la paz interior", 30],
    ["Das Schätzen von Momenten der Ruhe trägt zum emotionalen Gleichgewicht und inneren Frieden bei", 40],
    ["Investir tempo em autodesenvolvimento é um investimento valioso no nosso próprio potencial", 10],
    ["Investing time in self-development is a valuable investment in our own potential", 20],
    ["Invertir tiempo en autodesarrollo es una inversión valiosa en nuestro propio potencial", 30],
    ["Zeit in die eigene Weiterentwicklung zu investieren, ist eine wertvolle Investition in unser eigenes Potenzial",
     40],
    ["Aprender com os fracassos é uma estratégia essencial para o crescimento pessoal e profissional", 10],
    ["Learning from failures is an essential strategy for personal and professional growth", 20],
    ["Aprender de los fracasos es una estrategia esencial para el crecimiento personal y profesional", 30],
    ["Aus Fehlern zu lernen, ist eine wesentliche Strategie für persönliches und berufliches Wachstum", 40],
    ["Praticar a empatia fortalece os laços sociais e promove um entendimento mais profundo entre as pessoas", 10],
    ["Practicing empathy strengthens social bonds and promotes a deeper understanding among people", 20],
    ["Practicar la empatía fortalece los vínculos sociales y promueve una comprensión más profunda entre las personas",
     30],
    ["Empathie zu praktizieren stärkt soziale Bindungen und fördert ein tieferes Verständnis zwischen den Menschen",
     40],
    ["Desafiar-se regularmente leva ao crescimento pessoal, expandindo os limites do que é possível", 10],
    ["Challenging oneself regularly leads to personal growth, expanding the limits of what is possible", 20],
    ["Desafiarse regularmente conduce al crecimiento personal, expandiendo los límites de lo que es posible", 30],
    ["Sich regelmäßig herauszufordern führt zu persönlichem Wachstum und erweitert die Grenzen des Möglichen", 40],
    ["Cultivar gratidão diariamente é uma prática que eleva o espírito e promove o bem-estar emocional", 10],
    ["Cultivating gratitude daily is a practice that uplifts the spirit and promotes emotional well-being", 20],
    ["Cultivar la gratitud diariamente es una práctica que eleva el espíritu y promueve el bienestar emocional", 30],
    ["Dankbarkeit täglich zu pflegen ist eine Praxis, die den Geist erhebt und das emotionale Wohlbefinden fördert",
     40],
    ["Buscar oportunidades de aprendizado em cada experiência cotidiana é um hábito que enriquece a mente", 10],
    ["Seeking learning opportunities in every daily experience is a habit that enriches the mind", 20],
    ["Buscar oportunidades de aprendizaje en cada experiencia diaria es un hábito que enriquece la mente", 30],
    ["Lernmöglichkeiten in jeder täglichen Erfahrung zu suchen ist eine Gewohnheit, die den Geist bereichert", 40],
    ["A aceitação da mudança é fundamental para a adaptação constante e a evolução pessoal", 10],
    ["Acceptance of change is crucial for constant adaptation and personal evolution", 20],
    ["La aceptación del cambio es fundamental para la adaptación constante y la evolución personal", 30],
    ["Die Akzeptanz von Veränderung ist entscheidend für ständige Anpassung und persönliche Entwicklung", 40],
    ["Manter uma mente aberta para diferentes perspectivas amplia a compreensão do mundo ao nosso redor", 10],
    ["Keeping an open mind to different perspectives broadens our understanding of the world around us", 20],
    ["Mantener una mente abierta a diferentes perspectivas amplía nuestra comprensión del mundo que nos rodea", 30],
    ["Eine offene Haltung gegenüber verschiedenen Perspektiven erweitert unser Verständnis der Welt um uns herum", 40],
    ["A prática da resiliência fortalece a capacidade de superar desafios e se recuperar de adversidades", 10],
    ["Practicing resilience strengthens the ability to overcome challenges and recover from adversities", 20],
    ["La práctica de la resiliencia fortalece la capacidad para superar desafíos y recuperarse de adversidades", 30],
    [
        "Das Üben von Resilienz stärkt die Fähigkeit, Herausforderungen zu überwinden und sich von Widrigkeiten zu erholen",
        40]
]
def get_length(sublist):
    return sublist[1]
data = sorted(data, key=get_length)
# Tokenize the text data
tokenizer = Tokenizer()
texts = np.array([item[0] for item in data])
tokenizer.fit_on_texts(texts)


# Converte o texto em token, então por exemplo a palavra "is" pode ser o token 1
# Então toda vez que aparecer 1, significa que é a palavra "is"
sequences = tokenizer.texts_to_sequences(texts)
# print("Sequences")
# print(sequences)

# Cria arrays com tamanho máximo de 20, preenche com o número 0 nos espaços vazios
input_shape = 20
sequences = pad_sequences(sequences, maxlen=input_shape, padding='post', truncating='post')

# Example target data (random values for demonstration)
# target_data = np.random.randn(len(texts), 1)
results = np.array([item[1] for item in data])
print(results)
print(len(texts))

# Print shapes of generated data
# print("Padded Sequences Shape:", train_data.shape)
# print("Target Data Shape:", target_data.shape)

separacao = int(len(sequences)*0.8)
np.random.shuffle(sequences)
np.random.shuffle(results)

X_train, X_test = sequences[:separacao], sequences[separacao:]
y_train, y_test = results[:separacao], results[separacao:]

# ====== Primeiro modelo ========
# Define LSTM model
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, input_shape=(input_shape, 1)),
#     tf.keras.layers.Dense(1)
# ])
#
# # Compile model
# model.compile(loss='mse', optimizer='adam')
# Train Score: 121.57 MSE (11.03 RMSE)
# Test Score: 135.37 MSE (11.63 RMSE)

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(input_shape, 1)))
model.add(LSTM(units=64, return_sequences=True))
# model.add(LSTM(units=64, return_sequences=True))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
# Train Score: 125.21 MSE (11.19 RMSE)
# Test Score: 133.83 MSE (11.57 RMSE)

# trainScore = model.evaluate(X_train, y_train, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# testScore = model.evaluate(X_test, y_test, verbose=0)
# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))



# Train model
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=2)
# model.fit(train_texts, train_results, epochs=10)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


# teste = [["Cultivar gratidão diariamente é uma prática que eleva o espírito e promove o bem-estar emocional", 10]]
# teste = [["Lernmöglichkeiten in jeder täglichen Erfahrung zu suchen ist eine Gewohnheit, die den Geist bereichert", 40],
#          ["Das Üben von Resilienz stärkt die Fähigkeit, Herausforderungen zu überwinden und sich von Widrigkeiten zu erholen", 40]]
# test_X = np.array([item[0] for item in teste])
# test_Y = np.array([item[1] for item in teste])
# tokenizer.fit_on_texts(test_X)
# test_X = tokenizer.texts_to_sequences(test_X)
# test_X = pad_sequences(test_X, maxlen=input_shape, padding='post', truncating='post')
# testScore = model.evaluate(test_X, test_Y, verbose=0)
# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

