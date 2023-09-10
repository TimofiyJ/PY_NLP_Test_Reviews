import pandas as pd # dataframes and csv
import numpy as np # linear algebra
import sklearn as sk # data maniputation
import matplotlib.pyplot as plt # data visualization
import re # regular expressions for the text
import tensorflow as tf # ML
import sys # files managing
import csv # save result

from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

#arguments
reviews_location = str(sys.argv[1])
labels_location = str(sys.argv[2])

reviews = pd.read_csv(reviews_location)

sentences = []
ids = []
pattern = r'[,\s!?()._\-]+'
max_len=0

#data cleaning and splitting 
for i in range (len(reviews["text"])):
    sentence = reviews["text"][i]
    id = reviews["id"][i]
    sentence = str(sentence).lower()
    sentence = re.split(pattern, sentence) # Split the text based on the pattern
    if len(sentence)>max_len:
        max_len=len(sentence)
    sentence = " ".join(sentence)
    sentences.append(str(sentence))
    ids.append(id)

#downloading model and tokenizer
model = tf.keras.models.load_model('./my_additionally_trained_model')
tokenizer = BertTokenizerFast.from_pretrained('./my_additionally_trained_tokenizer')
results = []

#tokenizing each sentence and classificate it 
for i in range(len(sentences)):
    input_sentence = sentences[i]
    encoding = tokenizer(input_sentence, truncation=True, max_length=max_len, return_tensors='tf', pad_to_max_length=False, truncation_strategy='only_first')
    num_padding_tokens = max_len - encoding.input_ids.shape[-1]
    padding_tokens = tf.zeros((encoding.input_ids.shape[0], num_padding_tokens), dtype=tf.int32)
    padded_input_ids = tf.concat([padding_tokens, encoding.input_ids], axis=-1)
    input_sequence = padded_input_ids
    predicted_probability = model.predict(input_sequence)
    # Interpret the prediction (assuming binary classification)
    if predicted_probability > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    results.append({"id":ids[i],"sentiment":sentiment})

#write results
with open(labels_location, 'w',newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ["id","sentiment"])
    writer.writeheader()
    writer.writerows(results)
   

