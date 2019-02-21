import os
import json
import string
import numpy as np
from bert_serving.client import BertClient
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle

bc = BertClient(check_version=False, check_length=False)
stop = stopwords.words('english') + list(string.punctuation)
ontology = json.load(open("/nas/Datasets/dstc/dstc2/traindev/scripts/config/ontology_dstc2.json"))

dataset_path = "/nas/Datasets/dstc/dstc2/traindev/data/"
label_files = []


def is_word_in_ontology(word, slot_type="food"):
    pricerange_options = ontology["informable"]["pricerange"]
    food_options = ontology["informable"]["food"]
    area_options = ontology["informable"]["area"]

    if slot_type == "food":
        pass
    elif slot_type == "area":
        pass
    else:
        pass


for root, subdirs, files in os.walk(dataset_path):
    if len(files) != 0:
        label_files.append(f"{root}/label.json")

with open("train_data_food.csv", "w+") as output_file:
    for label_file_path in label_files:
        label_file = json.load(open(label_file_path))

        for turn in label_file['turns']:
            user_utterance = turn["transcription"]
            tokenized_utterance = word_tokenize(user_utterance)
            processed_utterance = [word for word in tokenized_utterance if word not in stop]

            sentence_embedding = np.array(bc.encode([user_utterance]))
            # word_embeddings = np.array(bc.encode(tokenized_utterance))
            for word in tokenized_utterance:
                word_embedding = np.array(bc.encode([word]))
                word_in_ontology = is_word_in_ontology(word, slot_type="food")

            # label = 0
            #
            # data = {
            #     "sentence_embeddings": sentence_embedding,
            #     "word_embeddings": word_embeddings
            # }
            #
            # pickle.dump(data, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.load()

        # for turn in label_file["turns"]:
        #
        #     user_uterrance = turn["transcription"].replace(',', '')
        #     labels = []
        #     for semantic_json in turn["semantics"]["json"]:
        #         if semantic_json["act"] == "inform":
        #
        #             slot, value = semantic_json["slots"][0]
        #             if slot != "this":
        #                 labels.append(f"{slot}={value}")
        #     if labels:
        #         output_file.write(f"{user_uterrance},{';'.join(labels)}\n")
