import copy
import pickle
import string
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from bert_serving.client import BertClient
from sklearn.model_selection import StratifiedKFold
from trackers.abstract_tracker import AbstractTracker
from contextualbandits.online import LinUCB


class BanditTracker(AbstractTracker):

    def __init__(self, ontology):
        super(BanditTracker, self).__init__(ontology)
        self.bc = BertClient(check_version=False, check_length=False)
        self.food_classifier, self.area_classifier, self.price_classifier = self.train()

    def is_word_in_ontology(self, word, slot_type="food"):
        """
        Returns a boolean saying whether a given word is present in the ontology
        :param word: The word to check if it's in the ontology
        :param slot_type: The type of slot in which to check if the word is present
        :return: a Boolean saying whether a given word is present in the ontology
        """

        if slot_type == "food":
            return int(word in self.ontology["informable"]["food"])

        elif slot_type == "area":
            return int(word in self.ontology["informable"]["area"])

        else:
            return int(word in self.ontology["informable"]["pricerange"])

    def train(self):
        food_training_data = pickle.load(open("../training_data/train_data_food_v2", "rb"))
        area_training_data = pickle.load(open("../training_data/train_data_area_v2", "rb"))
        price_training_data = pickle.load(open("../training_data/train_data_pricerange_v2", "rb"))

        # Model Instantiation
        food_classifier = area_classifier = price_classifier = LinUCB(2)

        print("training food")
        food_classifier.fit(np.array(food_training_data["features"]), np.array(food_training_data["labels"]),
                            np.array(food_training_data["labels"]))

        print("training area")
        area_classifier.fit(np.array(area_training_data["features"]), np.array(area_training_data["labels"]),
                            np.array(area_training_data["labels"]))

        print("training price")
        price_classifier.fit(np.array(price_training_data["features"]), np.array(price_training_data["labels"]),
                             np.array(price_training_data["labels"]))

        return food_classifier, area_classifier, price_classifier

    def addTurn(self, turn):
        """
        Adds a turn to this tracker
        :param turn: The turn to process and add
        :return: A hypothesis of the current state of the dialog
        """

        hyps = copy.deepcopy(self.hyps)

        goal_stats = defaultdict(lambda: defaultdict(float))

        # Obtaining the best hypothesis from the ASR module
        best_asr_hyp = turn['input']["live"]['asr-hyps'][0]["asr-hyp"]

        # English stopwords set with punctuation
        stop = stopwords.words('english') + list(string.punctuation)

        # Tokenize the best hypothesis on the whitespaces
        tkns = word_tokenize(best_asr_hyp)

        # Remove stop words and also shingle the tokens
        processed_hyp = [word for word in tkns if
                         word not in stop]  # + [tup[0] + " " + tup[1] for tup in ngrams(tkns, 2)]

        # Manually change from "moderately"/"affordable" to "moderate" and "cheaper" to "cheap"
        for idx, word in enumerate(processed_hyp):
            if word == "moderately" or word == "affordable":
                processed_hyp[idx] = "moderate"
            if word == "cheaper":
                processed_hyp[idx] = "cheap"

        if processed_hyp:

            # Create an embedding of the user utterance using BERT
            sentence_embedding = np.array(self.bc.encode([best_asr_hyp]))[0]

            # Iterate through all the words in the user utterance to obtain the features needed
            for word in processed_hyp:
                # Create and embedding of the word, being iterated, using BERT
                word_embedding = np.array(self.bc.encode([word]))[0]

                # Check whether the current word is present in the ontology, in one of the slot types
                word_in_food_ontology = [self.is_word_in_ontology(word, slot_type="food")]
                word_in_area_ontology = [self.is_word_in_ontology(word, slot_type="area")]
                word_in_price_ontology = [self.is_word_in_ontology(word, slot_type="price")]

                # Concatenate the features together (the result is a vector of size 2049)
                food_features = np.concatenate((word_embedding, sentence_embedding, word_in_food_ontology))
                area_features = np.concatenate((word_embedding, sentence_embedding, word_in_area_ontology))
                price_features = np.concatenate((word_embedding, sentence_embedding, word_in_price_ontology))

                # Decide whether the current word should update one (or more) of the slot types
                update_food_slot = self.food_classifier.predict([food_features])[0]
                update_area_slot = self.area_classifier.predict([area_features])[0]
                update_price_slot = self.price_classifier.predict([price_features])[0]

        self.hyps = hyps
        return self.hyps
