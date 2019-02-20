import numpy as np
from bert_serving.client import BertClient
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing


def map_function(line):
    feature, value = line.split(",")
    return feature, value.replace('\n', '')

#Abandoning this idea

class BertClassifierTracker:

    def __init__(self):
        self.data = self.read_data_file()

    def read_data_file(self):
        f = open("train_data.csv", "r")
        lines = f.readlines()
        mapped_lines = list(map(lambda line: map_function(line), lines))
        features = map(lambda line: line[0], mapped_lines)
        labels = map(lambda line: line[1], mapped_lines)
        label_encoder = preprocessing.LabelEncoder()

        # encoded_labels =

        # for line in lines:
        #     feature, value = line.split(',')
        #     # encoded_feature

        def process_data(self, data):
            pass
