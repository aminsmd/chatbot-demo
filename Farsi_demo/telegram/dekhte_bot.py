# imports
# from tensorflow import keras
import keras
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Bidirectional
from nltk.tokenize.treebank import TreebankWordTokenizer
from scipy import spatial
from random import shuffle
import pickle
from keras.callbacks import ModelCheckpoint
from scipy import spatial
import csv
import requests
import json
from textblob import TextBlob


def prepareSent(sent):
    #global embeddings_index
    marks = {'؟', '!', '.', '،'}
    zamir_1 = {'م', 'ش', 'ت', 'ه', 'و', 'ی'}
    zamir_3 = {'تان', 'شان', 'مان'}
    sent = sent.split()
    sent = [i[:-1] if i[-1] in marks or (
                i[-1] in zamir_1 and i[:-1] in embeddings_index and i not in embeddings_index) else i for i in sent]
    sent = [i[:-3] if i[-3:] in zamir_3 and i[:-3] in embeddings_index and i not in embeddings_index else i for i in
            sent]
    tokenized = []
    n = 0
    while (n != len(sent)):
        if n < len(sent) - 2 and (sent[n] + "‌" + sent[n + 1] + "‌" + sent[n + 2]) in embeddings_index:
            tokenized.append(sent[n] + "‌" + sent[n + 1] + "‌" + sent[n + 2])
            n += 2
        elif n != len(sent) - 1 and (sent[n] + "‌" + sent[n + 1]) in embeddings_index:
            tokenized.append(sent[n] + "‌" + sent[n + 1])
            n += 1
        else:
            tokenized.append(sent[n])
        n += 1
    return tokenized


# recognize entity
def return_entity(sent, entity):
    #global embeddings_index
    sent = sent.lower()
    ma = 0
    ans = ""
    for i in sent.split():
        if sim(embeddings_index[i], entity) > ma:
            ma = sim(embeddings_index[i], entity)
            ans = i
    if ma < .35:
        return "nothing"
    return ans


def wordToVec(data):
    MAX_SEQ = 20
    #global MAX_SEQ
    for s in range(len(data)):
        n = MAX_SEQ - len(data[s])
        if n < 0:
            data[s] = data[s][:MAX_SEQ]
        else:
            for i in range(n):
                data[s].append('<PAD>')
        for v in range(len(data[s])):
            if data[s][v] not in embeddings_index:
                data[s][v] = embeddings_index['<UNK>']
            else:
                data[s][v] = embeddings_index[data[s][v]]
    return np.array(data)


def sim(dataSetI, dataSetII):
    return 1 - spatial.distance.cosine(dataSetI, dataSetII)


# defining entities
def initalize():
    MAX_SEQ = 20

    global farsi_model
    global embeddings_index
    global entity_lists

    #global MAX_SEQ
    marks = {'؟', '!', '.', '،'}
    zamir_1 = {'م', 'ش', 'ت', 'ه', 'و', 'ی'}
    zamir_3 = {'تان', 'شان', 'مان'}
    # global stop_words

    # stop_words = set(stopwords.words('english'))
    # stop_words.add('?')
    # stop_words.remove('now')
    with open("farsi embeddings", 'rb') as fp:
        embeddings_index = pickle.load(fp)



    # defining entities
    embeddings_size = 300
    entity_lists = {"cloth": ['شلوار', 'کاپشن', 'پیراهن'],
                    "city_iran": ['شیراز', 'کرج', 'مشهد'],
                    "food": ['کباب', 'ساندویچ', 'سوپ'],
                    "time": ['امروز', 'فردا', 'دیروز']
                    }
    for ent in entity_lists:
        sum_of_embedding = np.zeros(embeddings_size)
        for obj in entity_lists[ent]:
            sum_of_embedding += np.array(embeddings_index[obj])
        sum_of_embedding /= len(entity_lists[ent])
        globals()['entity_{}'.format(ent)] = list(sum_of_embedding)

    # reading embedding_dict
    class_size = 5
    # building model
    input_layer = Input(batch_shape=(None, MAX_SEQ, 300))
    lstm_layer = Bidirectional(LSTM(units=MAX_SEQ))(input_layer)
    output_layer = Dense(class_size, activation="softmax")(lstm_layer)

    farsi_model = Model(inputs=input_layer, outputs=output_layer)
    farsi_model.compile(loss='categorical_crossentropy',
                        optimizer='adam')
    #farsi_model.summary()
    # load previous weights
    farsi_model.load_weights('weight_farsi_dekhte.50.hdf5')


#   print(sim(embeddings_index['now'],entity_time))


def classify(sent):
    class_size = 5
    sentence = prepareSent(sent)
    sent = ' '.join(list(sentence))
    sentence = wordToVec([sentence])[0]
    sentence = np.reshape(sentence, (1, 20, 300))

    sentence = farsi_model.predict(sentence)
    argmax = np.argmax(sentence)
    a = [1 if argmax == i else 0 for i in range(class_size)]
    res = whichClass(list(a))
    if res == 'Address':
        return address(sent)
    elif res == 'Weather':
        return weather(sent)
    elif res == 'Restaurant':
        return food(sent)
    else:
        return res


def whichClass(inp):
    for i in range(len(inp)):
        if 1 == inp[i]:
            if i == 0:
                return "Address"
            elif i == 1:
                return "Restaurant"
            elif i == 2:
                return ("الآن به همکارام تو خانه‌داری اطلاع میدم")
            elif i == 3:
                return ("الآن به همکارام تو خشک‌شویی اطلاع میدم")
            elif i == 4:
                return "Weather"


def address(sent):
    sent = sent.replace(' ', '+')
    res = requests.get("https://www.google.com/maps/search/?api=1&query={}+اصفهان&hl=fa".format(sent))
    return (res.url)


def food(sent):
    menu = ['منو', 'فهرست', 'منوی', 'لیست']
    spl = sent.split()
    for i in menu:
        if i in spl:
            print('! الآن منو رو برات میفرستم')
            return
    return ('الآن سفارشتونو به همکارام میگم')


def weather(sent):
    city = return_entity(sent, entity_city_iran)
    if city == 'nothing':
        city = 'اصفهان'
    blob = TextBlob(city)
    blob = blob.translate(to="en")
    res = requests.get(
        "https://api.openweathermap.org/data/2.5/weather?q={}&appid=3bb1d3931ea6593a0833bd5cf0b97ac3&lang=fa".format(
            city))
    if res.status_code == 200:
        data = json.loads(res.text)
        temp = int(data['main']['temp'] - 273.15)
        desc = data['weather'][0]['description']
        blob = TextBlob(desc)
        print('{} : {}'.format(city, desc))
        return ('و دمای هوا {} درجه سانتیگراد می‌باشد'.format(temp))
    else:
        print(res.text)
