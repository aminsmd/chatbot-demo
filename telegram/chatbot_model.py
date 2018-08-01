#imports 
import keras
import numpy as np
from keras.models import Model
from keras.layers import LSTM,Dense,Input,Bidirectional
from nltk.tokenize.treebank import TreebankWordTokenizer
from scipy import spatial
import pickle


def prepare(sentence):

    MAX_SEQ = 20
    tokenizer = TreebankWordTokenizer()
    sent = tokenizer.tokenize(sentence)
    for i in sent:
        n = MAX_SEQ - len(sent)
    if n < 0:
      sent = sent[:MAX_SEQ]
    else:
        for j in range(n):
            sent.append('<PAD>')
    for j in range(len(sent)):
        if sent[j] in embeddings_index:
            sent[j] = embeddings_index[sent[j]]
        else:
            sent[j] = embeddings_index["<UNK>"]
    return np.array(sent).reshape((1, 20, 300))

#recognize entity
def return_entity(sent , entity):
    sent = sent.lower()
    ma = 0
    ans = ""
    for i in sent.split():
        if sim(embeddings_index[i] , entity) > ma:
            ma = sim(embeddings_index[i] , entity)
            ans = i
    if ma < .1:
        return "nothing"
    return ans

def classify(sent):
    sentence = prepare(sent)
    sentence = dekhtemodel.predict(sentence)
    argmax = np.argmax(sentence)
    if argmax == 0:
        sentt='AddToPlaylist'
    elif argmax == 1:
        sentt='BookRestaurant'
    elif argmax == 2:
        sentt='GetWeather'
    elif argmax == 3:
        sentt='PlayMusic'
    elif argmax == 4:
        sentt='RateBook'
    elif argmax == 5:
        sentt='SearchCreativeWork'
    elif argmax == 6:
        sentt='SearchScreeningEvent'

    print(sentt)
    return sentt

#similarity function
def sim(dataSetI , dataSetII):
    return 1 - spatial.distance.cosine(dataSetI, dataSetII)

#defining entities
def initalize():


    global dekhtemodel
    global embeddings_index


    with open("embedding_dict", 'rb') as fp:
        embeddings_index = pickle.load(fp)


    embeddings_size = 300
    entity_lists = {"cloth": ['t-shirt', 'shirts', 'jeans'],
                    "city_iran": ['karaj', 'tehran', 'mashhad'],
                    "name_foreign": ['john', 'jack', 'paul'],
                    "music_genre": ['pop', 'rap', 'jazz', 'rock', 'classical'],
                    "time": ['tommorow', 'today', 'yesterday', '8pm'],
                    "adverb": ['sometimes', 'usually', 'never']
                    }
    for ent in entity_lists:
        sum_of_embedding = np.zeros(embeddings_size)
        for obj in entity_lists[ent]:
            sum_of_embedding += np.array(embeddings_index[obj])
        sum_of_embedding /= len(entity_lists[ent])
        globals()['entity_{}'.format(ent)] = list(sum_of_embedding)

    # reading embedding_dict

    # building model
    BATCH_SIZE = 1
    MAX_SEQ = 20
    input_layer = Input(batch_shape=(BATCH_SIZE, MAX_SEQ, 300))
    lstm_layer = Bidirectional(LSTM(units=MAX_SEQ))(input_layer)
    output_layer = Dense(7, activation="softmax")(lstm_layer)

    dekhtemodel = Model(inputs=input_layer, outputs=output_layer)
    dekhtemodel.compile(loss='categorical_crossentropy',
                        optimizer='adam')

    # load previous weights
    dekhtemodel.load_weights('weight_dekhte.12.hdf5')
