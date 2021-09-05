# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 10:24:23 2021

@author: Charles Arthel Rey
"""


file  = r"C:\Users\Charles Arthel Rey\Downloads\keppel-corporation-limited-annual-report-2018.pdf"

""" Assignment # 1 """

import fitz
from openpyxl import Workbook

def stringToList(string):
    listRes = list(string.split("\n\n"))
    return listRes

#extract text page by page
doc = fitz.open(file)
page = doc.load_page(11)
pymupdf_text = page.getText()
pymupdf_text = pymupdf_text.replace(". \n", ". \n\n") # this code breaks the sentences per line
pymupdf_text = pymupdf_text.replace(".\n", ". \n\n")

jotdown = ["\n\nStarting", "\n\nWith the burgeoning", "\n\nUrbanFox", "\n\nKeppel Infrastructure Trust", "\n\nWe continued to drive", "\n\nWith\xa0the Eco-City’s"]


# this code will join the sentences belonging to a single paragraph, but were wrongfully separated into two paragraphs
for item in jotdown:
    pymupdf_text = pymupdf_text.replace(item, item[2:])

# this code deletes this specific line    
pymupdf_text = pymupdf_text.replace("\n1\nKeppel Land expanded \nits presence in China \nin 2018 entering a new \nmarket with\xa0a residential \nland\xa0plot\xa0in Nanjing. \n\n2\nKeppel O&M’s proprietary \nRigCare Solution, \nimplemented for the \nfirst\xa0time on Cantarell IV, \nwill enhance the efficiency, \nsafety and operability of \nthe jackup rig. \n\n", "")

#this code initializes the excel file
wb = Workbook()
sheet = wb.active
# this code divides the string into a list where each item is a paragraph
strings = stringToList(pymupdf_text)
for i in range(len(strings)):
    strings[i] = strings[i].replace(" \n", " ")
    c = sheet.cell(row = i+1 , column = 1)
    c.value = strings[i]

#print("\n\n".join(strings))
wb.save('Rey,Charles_PythonAssign1.xlsx')




""" Assignment # 2 """

import tabula

table = tabula.read_pdf(file, pages=69)
table = (table[0])


"""table1 = table.iloc[:1]
table2 = table.iloc[1:]
list_un2 = list(table['Unnamed: 2'])

for i in range(1, len(list_un2)):
     list_un2[i]= list_un2[i].replace(" ", ",")
"""
     
table[['Nomination' ,'Remuneration']] = table['Unnamed: 2'].str.split(" ", 1, expand=True)

table['Unnamed: 2'] = table['Nomination']
table.insert(4, 'Remuneration2', table['Remuneration'])
table.drop(columns = ['Nomination','Remuneration'], axis="columns", inplace=True)

table.columns = [" " if i > 1 else x for i, x in enumerate(table.columns, 1)]


table.to_excel("Rey,Charles_PythonAssign2.xlsx")






""" Assignment # 3 """


# dataset preprocessing

corpus = open(r"C:\Users\Charles Arthel Rey\Downloads\restauranttrain.bio.txt")
trans_corpus = ""
# this code extracts text from text file, corpus.
for line in corpus:
    trans_corpus = trans_corpus + line

# this code divides the string file per sentence 
trans_corpus = list(trans_corpus.split("\n\n")) # the goal is to make each item in this list equivalent to one sentence.   
ftrans_corpus = []
for i in range(len(trans_corpus)):
    trans_corpus[i] = list(trans_corpus[i].split("\n"))
    tups = []
    for item in trans_corpus[i]:
        tups.append(tuple(item.split("\t")))
    
    trans_corpus[i] = (tups)

trans_corpus = [ele for ele in trans_corpus if len(ele) > 1]

sentences = []
all_entities = []
for grplist in trans_corpus:
    single_sentence = []
    single_entities = []
    for tups in grplist:
        single_sentence.append(tups[1])
        if tups[0] == 'O':
            del tups
        else:
            single_entities.append(tups)
    
    sentences.append(" ".join(single_sentence))
    all_entities.append(single_entities)
    
Train_Entities = []
for i in range(len(all_entities)):
    entities = []
    for tuples in all_entities[i]:
        index = (sentences[i].index(tuples[1]), sentences[i].index(tuples[1])+len(tuples[1]), tuples[0])
        entities.append(index)

    Train_Entities.append(entities)

# this code aims to delete all items has no entities in its sentence

all_entities = [ele for ele in all_entities if Train_Entities[all_entities.index(ele)] !=  []]
sentences = [ele for ele in sentences if Train_Entities[sentences.index(ele)] !=  []]
Train_Entities =  [ele for ele in Train_Entities if ele != []] 

TRAIN_DATA = []
for i in range(len(Train_Entities)):
    train_tups = (sentences[i], {'entities': Train_Entities[i]})
    TRAIN_DATA.append(train_tups)
    
""" TRAIN_DATA now follows the format of the dataset required to be fed to Spacy. """


import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
import nltk
from sklearn.model_selection import train_test_split



TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")
BERT = TFAutoModel.from_pretrained("bert-base-cased")


BERT_TRAIN_DATA = [x for l in trans_corpus for x in l]
tags0 = dict(enumerate((set([ele[0] for ele in BERT_TRAIN_DATA]))))
tags = {v: k for k, v in tags0.items()}

train_ds, test_ds = train_test_split(BERT_TRAIN_DATA, test_size=0.20, random_state=42)


SEQ_LEN = 20

def tokenize(sentence):
    
    tokens = TOKENIZER.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    tokenss = TOKENIZER.tokenize(sentence)
    return tokens['input_ids'], tokens['attention_mask'], tokenss

def INITIALIZE(_ds):
    all_text = []
    all_tags = []
    for tupl in _ds:
        all_text.append(tupl[1])
        all_tags.append(tupl[0])

    all_tags = [tags[x] for x in all_tags]

    arr = np.array(all_tags)  # take sentiment column in df as array
    labels = np.zeros((arr.size, arr.max()+1))  # initialize empty (all zero) label array
    labels[np.arange(arr.size), arr] = 1  # add ones in indices where we have a value
    
    # initialize two arrays for input tensors
    Xids = np.zeros((len(all_text), SEQ_LEN))
    Xmask = np.zeros((len(all_text), SEQ_LEN))
    sentence_tokens = []

    # loop through data and tokenize everything
    for i, sentence in enumerate(all_text):
        Xids[i, :], Xmask[i, :], tokens = tokenize(sentence)
        sentence_tokens.append(tokens)

    # create tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    # restructure dataset format for BERT
    def map_func(input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_mask': masks}, labels
  
    dataset  = dataset.map(map_func)  # apply the mapping function
    
    return dataset.shuffle(10000).batch(32)

dataset_train = INITIALIZE(train_ds)
dataset_test = INITIALIZE(test_ds)

# build the model
input_ids = tf.keras.layers.Input(shape=(50,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(50,), name='attention_mask', dtype='int32')

embeddings = BERT(input_ids, attention_mask=mask)[0]  # we only keep tensor 0 (last_hidden_state)

X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
y = tf.keras.layers.Dense(17, activation='softmax', name='outputs')(X)  # adjust based on number of sentiment classes

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

# freeze the BERT layer
model.layers[2].trainable = False

# compile the model
optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

history = model.fit(dataset_train, epochs = 1)

#history = model.fit({'input_ids':Xids_train, 'attention_mask': Xmask_train}, Xlabels_train, epochs =1)

test_loss, test_acc = model.evaluate(dataset_test, verbose = 2)

# this code will predict the entities when inputting a text. 

def Prediction(text):
    predicts = nltk.word_tokenize(text)

    # initialize two arrays for input tensors
    Xids_pred = np.zeros((len(predicts), SEQ_LEN))
    Xmask_pred = np.zeros((len(predicts), SEQ_LEN))
    sentence_tokens_pred = []

    # loop through data and tokenize everything
    for i, sentence in enumerate(predicts):
        Xids_pred[i, :], Xmask_pred[i, :], tokens_pred = tokenize(sentence)
        sentence_tokens_pred.append(tokens_pred)
    
    pred = model.predict({'input_ids':Xids_pred, 'attention_mask': Xmask_pred})
    pred = np.argmax(pred, axis = 1)
    results = []
    for item in pred:
        results.append(tags0[item])
    
    return results


pred = Prediction("During quanrantine, it is now allowed to dine inside Jollibee")
print(pred)