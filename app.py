# Library imports
import spacy
import string
import pandas as pd
import nltk
import re
import numpy as np

from tensorflow import keras
from nltk.corpus import wordnet as wn
from numpy import array
from keras.models import load_model
from flask import Flask, request, jsonify, render_template
from numpy import asarray, zeros
from numpy import zeros
from copy import deepcopy

embeddings_dictionary = dict()
glove_file = open('glove.6B/glove.6B.300d.txt', encoding="utf8")

for line in glove_file:
  records = line.split()
  word = records[0]
  vector_dimensions = asarray(records[1:], dtype='float32')
  embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# Load trained Pipeline
model = load_model('model.keras')
nlp = spacy.load('en_core_web_lg')

# Create the app object
app = Flask(__name__)

# Search for the first antonym for a word to handle negations
def antonym_for(word):
  antonyms = set()
  for ss in wn.synsets(word):
    for lemma in ss.lemmas():
      any_pos_antonyms = [ antonym.name() for antonym in lemma.antonyms() ]
      for antonym in any_pos_antonyms:
        antonym_synsets = wn.synsets(antonym)
        if wn.ADJ not in [ ss.pos() for ss in antonym_synsets ]:
          continue
        return antonym

  return antonyms

# Remove punctuations, handle negation and Tokenize
def tokenize_review(input_string):
  doc = nlp(input_string)
  doc_len = len(doc)
  skip_next_word = False
  tokens = []

  for i, token in enumerate(doc):
    if(skip_next_word):
      skip_next_word = False
      continue

    if(token.dep_ == "neg" and i < (doc_len - 1) and doc[i + 1].pos_ == "ADJ"):
      antonym = antonym_for(doc[i + 1].lemma_)

      if(antonym != set()):
        skip_next_word = True
        tokens.append(antonym_for(doc[i + 1].lemma_))
      else:
        tokens.append(token)
    else:
      tokens.append(token.lemma_)

  return tokens

def message_to_word_vectors(review, word_dict=embeddings_dictionary):
  processed_list_of_tokens = tokenize_review(review)
  vectors = []

  for token in processed_list_of_tokens:
    if token not in word_dict:
      continue

    token_vector = word_dict[token]
    vectors.append(token_vector)

  return np.array(vectors).astype(float)

def preprocess_reviews(reviews):
  return [message_to_word_vectors(review) for review in reviews]

def pad_X(X, desired_sequence_length=300):
  X_copy = deepcopy(X)

  for i, x in enumerate(X):
    x_seq_len = x.shape[0]
    sequence_length_difference = desired_sequence_length - x_seq_len
    if sequence_length_difference < 0:
      X_copy[i] = x[:300]
    else:
      pad = np.zeros(shape=(sequence_length_difference, 300))
      X_copy[i] = np.concatenate([x, pad])

  return np.array(X_copy).astype(float)

# Define predict function
@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_string = request.form.get("input")
  word_vectors = [message_to_word_vectors(input_string)]
  X = pad_X(word_vectors)

  prediction = model.predict(X)[0][0]

  if(prediction < 0.5):
    return render_template('index.html', input_string=input_string, prediction_text='Negative', score=prediction)
  else:
    return render_template('index.html', input_string=input_string, prediction_text='Positive', score=prediction)


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000, debug=True)
