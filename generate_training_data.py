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
    if slot_type == "food":
        return int(word in ontology["informable"]["food"])

    elif slot_type == "area":
        return int(word in ontology["informable"]["area"])

    else:
        return int(word in ontology["informable"]["pricerange"])


def get_label(word, turn_obj, slot_type="food"):
    ground_truth_arr = turn_obj["semantics"]["json"]

    for ground_truth_obj in ground_truth_arr:
        slots_arr = ground_truth_obj["slots"]

        if len(slots_arr) > 0:
            slot, value = slots_arr[0]

            if slot == slot_type:
                return int(word == value)

    return 0


for root, subdirs, files in os.walk(dataset_path):
    if len(files) != 0:
        label_files.append(f"{root}/label.json")


def create_trainig_data(output_dict, slot_type, output_file_name):
    for label_file_path in label_files:
        label_file = json.load(open(label_file_path))

        for turn in label_file['turns']:
            user_utterance = turn["transcription"]
            tokenized_utterance = word_tokenize(user_utterance)
            # processed_utterance = [word for word in tokenized_utterance if word not in stop]

            sentence_embedding = np.array(bc.encode([user_utterance]))[0]

            for word in tokenized_utterance:

                if word == "moderately":
                    word = "moderate"

                if word == "cheaper":
                    word = "cheap"

                word_embedding = np.array(bc.encode([word]))[0]
                word_in_ontology = [is_word_in_ontology(word, slot_type=slot_type)]

                features = np.concatenate((word_embedding, sentence_embedding, word_in_ontology))

                label = get_label(word, turn, slot_type=slot_type)

                output_dict["features"].append(features)
                output_dict["labels"].append(label)

    pickle.dump(output_dict, open(output_file_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


# food_data = {
#     "word_embedding": [],
#     "sentence_embedding": [],
#     "word_in_ontology": [],
#     "label": []
# }

food_data = {
    "features": [],
    "labels": []
}

# area_data = {
#     "word_embedding": [],
#     "sentence_embedding": [],
#     "word_in_ontology": [],
#     "label": []
# }

area_data = {
    "features": [],
    "labels": []
}

# price_data = {
#     "word_embedding": [],
#     "sentence_embedding": [],
#     "word_in_ontology": [],
#     "label": []
# }

price_data = {
    "features": [],
    "labels": []
}

create_trainig_data(food_data, "food", "train_data_food_v2")
create_trainig_data(area_data, "area", "train_data_area_v2")
create_trainig_data(price_data, "pricerange", "train_data_pricerange_v2")
