import os
import json
import string
import numpy as np
from bert_serving.client import BertClient
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle

# Create BERT Client instance
bc = BertClient(check_version=False, check_length=False)

# English stopwords set and punctuation to be removed
stop = stopwords.words('english') + list(string.punctuation)

# A json object containing the ontology of the dataset
ontology = json.load(open("/nas/Datasets/dstc/dstc2/traindev/scripts/config/ontology_dstc2.json"))

dataset_path = "/nas/Datasets/dstc/dstc2/traindev/data/"
label_files = []


def is_word_in_ontology(word, slot_type="food"):
    """
    Returns a boolean saying whether a given word is present in the ontology
    :param word: The word to check if it's in the ontology
    :param slot_type: The type of slot in which to check if the word is present
    :return: a Boolean saying whether a given word is present in the ontology
    """
    if slot_type == "food":
        return int(word in ontology["informable"]["food"])

    elif slot_type == "area":
        return int(word in ontology["informable"]["area"])

    else:
        return int(word in ontology["informable"]["pricerange"])


def get_label(word, turn_obj, slot_type="food"):
    """
    Returns whether "word" is present the ground truth or not.
    :param word: The word to search for in the ground truth
    :param turn_obj: the object that contains this turns ground truth
    :param slot_type: the type of slot the classifier identifies
    :return: Returns 1 if "word" is in the ground truth, 0 otherwise
    """
    ground_truth_arr = turn_obj["semantics"]["json"]

    for ground_truth_obj in ground_truth_arr:
        slots_arr = ground_truth_obj["slots"]

        if len(slots_arr) > 0:
            slot, value = slots_arr[0]

            if slot == slot_type:
                return int(word == value)

    return 0


# Get all the training "label.json" files
for root, subdirs, files in os.walk(dataset_path):
    if len(files) != 0:
        label_files.append(f"{root}/label.json")


def create_trainig_data(slot_type, output_file_name):
    """
    Serializes a dictionary containing training features and labels for the given "slot_type"
    :param slot_type: The type of slot the classifier must identify
    :param output_file_name: The name of the file in which the serialized object is stored
    """
    # The format of the object to be serialized
    output_dict = {
        "features": [],
        "labels": []
    }

    # Iterate over every label.json file in the training/dev dataset
    for label_file_path in label_files:

        # Open the label.json file
        label_file = json.load(open(label_file_path))

        # Iterate over every "turn" in the label.json file
        for turn in label_file['turns']:

            user_utterance = turn["transcription"]
            tokenized_utterance = word_tokenize(user_utterance)
            # processed_utterance = [word for word in tokenized_utterance if word not in stop]

            # Create an embedding representation of the entire user utterance, using BERT, to use as a feature
            sentence_embedding = np.array(bc.encode([user_utterance]))[0]

            # For every word in the user sentence
            for word in tokenized_utterance:

                if word == "moderately" or word == "affordable":  # NOTE: "affordable" was not yet implemented when training set was created
                    word = "moderate"

                if word == "cheaper":
                    word = "cheap"

                # Create an embedding of the single word ,using BERT, to also use as a feature
                word_embedding = np.array(bc.encode([word]))[0]

                # Checks if the current word being iterated is present in the ontology, used as a feature
                word_in_ontology = [is_word_in_ontology(word, slot_type=slot_type)]

                # concatenate all the features together in a vector of size 2049
                features = np.concatenate((word_embedding, sentence_embedding, word_in_ontology))

                # Get the label associated with these features
                label = get_label(word, turn, slot_type=slot_type)

                output_dict["features"].append(features)
                output_dict["labels"].append(label)

    # Serialize training data dictionary
    pickle.dump(output_dict, open(output_file_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


create_trainig_data("food", "train_data_food_v2")
create_trainig_data("area", "train_data_area_v2")
create_trainig_data("pricerange", "train_data_pricerange_v2")
