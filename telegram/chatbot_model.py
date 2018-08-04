#imports 
import keras
import numpy as np
from keras.models import Model
from keras.layers import LSTM,Dense,Input,Bidirectional
from nltk.tokenize.treebank import TreebankWordTokenizer
from scipy import spatial
import pickle
from nltk.corpus import stopwords
import nltk


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
    global stop_words
    sent = sent.lower()
    tokenizer = TreebankWordTokenizer()
    sent = tokenizer.tokenize(sent)
    ma = 0
    ans = ""
    pos_tagged = nltk.pos_tag(sent)
    j = 0
    for i in sent:
        if i in embeddings_index:
            if i not in stop_words and pos_tagged[j][1][0:2] != "VB" and sim(embeddings_index[i] , entity) > ma:
                ma = sim(embeddings_index[i] , entity)
                ans = i
        else : print("not in embedding {}".format(i))
        j+=1
                

    if ma < .25 :
        return "nothing"
    return ans

def classify(sent):
    sentence = prepare(sent)
    sentence = dekhtemodel.predict(sentence)
    argmax = np.argmax(sentence)
    if argmax == 0:
        sentt='AddToPlaylist'
    elif argmax == 1:
        #sentt='BookRestaurant'
        nationality = return_entity(sent, entity_nationality)
        time = return_entity(sent, entity_time)
        if time != 'nothing' and nationality != 'nothing':
            sentt='you asked me too book you a {} restaurant for {}'.format(nationality, time)
        elif time == 'nothing' and nationality != 'nothing':
            sentt='when do you want to go there?'
        else:
            sentt='what kind of restaurant do you want?'
    elif argmax == 2:
        #sentt='GetWeather'
        city = return_entity(sent, entity_city_iran)
        time = return_entity(sent, entity_time)
        print(city)
        print(time)
        if time != 'nothing' and city != 'nothing':
            sentt='you requested {}\'s weather for {} ?'.format(city, time)
        elif time == 'nothing' and city != 'nothing':
            sentt='you requested {}\'s weather?'.format(city)
        else:
            sentt='which city\'s weather do you want to know ?'
        #print(sim(embeddings_index['karaj'], entity_city_iran))
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
    global stop_words

    stop_words = set(stopwords.words('english'))
    stop_words.add('?')
    stop_words.remove('now')
    with open("embedding_dict", 'rb') as fp:
        embeddings_index = pickle.load(fp)
    

    embeddings_size = 300
    entity_lists = {"cloth": ['t-shirt', 'shirts', 'jeans'],
                    "city_iran": ['karaj', 'tehran', 'mashhad'],
                    "name_foreign": ['john', 'jack', 'paul'],
                    "music_genre": ['pop', 'rap', 'jazz', 'rock', 'classical'],
                    "time": ['tommorow', 'today', 'yesterday', 'friday', 'saturdays', 'sunday', 'now'],
                    "adverb": ['sometimes', 'usually', 'never'],
                    "nationality": ['chinese', 'persian', 'french']
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

#   print(sim(embeddings_index['now'],entity_time))

