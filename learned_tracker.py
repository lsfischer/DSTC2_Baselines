import pickle
import string
import copy
import numpy as np
from sklearn import svm
from nltk import word_tokenize
from nltk.corpus import stopwords
from bert_serving.client import BertClient
from sklearn.model_selection import StratifiedKFold
from abstract_tracker import AbstractTracker
from collections import defaultdict


class LearnedTracker(AbstractTracker):

    def __init__(self, ontology):
        super(LearnedTracker, self).__init__(ontology)
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

    def train_aux(self, data_object):
        # convert to np_array
        data_object["features"] = np.array(data_object["features"])
        data_object["labels"] = np.array(data_object["labels"])

        cs = np.arange(1, 1000, 10)

        kfolds = StratifiedKFold(n_splits=5)

        cross_err_list = []  # To be converted into a matrix to be plotted with the c value, the training error and the validation error
        for c in cs:
            total_train_error = total_val_error = 0

            # Stratified k folds
            for train_idx, valid_idx in kfolds.split(data_object["labels"], data_object["labels"]):
                # Obtain the training and validation folds from the training set

                x_training_set = data_object["features"][train_idx]
                x_validation_set = data_object["features"][valid_idx]

                y_training_set = data_object["labels"][train_idx]
                y_validation_set = data_object["labels"][valid_idx]

                model = svm.SVC(kernel='rbf', C=c, gamma="auto")
                model.fit(x_training_set, y_training_set)
                training_error = 1 - model.score(x_training_set, y_training_set)
                validation_error = 1 - model.score(x_validation_set, y_validation_set)
                total_train_error += training_error
                total_val_error += validation_error

            cross_err_list.append((c, total_train_error, total_val_error))

        cross_err_matrix = np.array(cross_err_list)  # Convert error list into matrix form

        # find the best C value
        index_line_of_best_val = np.argmin(cross_err_matrix[:, 2])
        return cross_err_matrix[index_line_of_best_val, 0]

    def train(self):
        food_training_data = pickle.load(open("train_data_food_v2", "rb"))
        area_training_data = pickle.load(open("train_data_area_v2", "rb"))
        price_training_data = pickle.load(open("train_data_pricerange_v2", "rb"))

        print("training food")
        best_food_c = 131.0  # self.train_aux(food_training_data)
        food_classifier = svm.SVC(kernel="rbf", C=best_food_c, gamma="auto")
        food_classifier.fit(np.array(food_training_data["features"]), np.array(food_training_data["labels"]))

        print("training area")
        best_area_c = 51.0  # self.train_aux(area_training_data)
        area_classifier = svm.SVC(kernel="rbf", C=best_area_c, gamma="auto")
        area_classifier.fit(np.array(area_training_data["features"]), np.array(area_training_data["labels"]))

        print("training price")
        best_price_c = 1.0  # self.train_aux(price_training_data)
        price_classifier = svm.SVC(kernel="rbf", C=best_price_c, gamma="auto")
        price_classifier.fit(np.array(price_training_data["features"]), np.array(price_training_data["labels"]))

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

                # Predict the probability associated with each class
                # food_proba = self.food_classifier.predict_proba([food_features])[0]
                # area_proba = self.area_classifier.predict_proba([area_features])[0]
                # price_proba = self.price_classifier.predict_proba([price_features])[0]

                # Decide whether the current word should update one (or more) of the slot types
                update_food_slot = self.food_classifier.predict([food_features])[0]
                update_area_slot = self.area_classifier.predict([area_features])[0]
                update_price_slot = self.price_classifier.predict([price_features])[0]

                if update_food_slot:
                    goal_stats["food"][word] += 1.0

                if update_area_slot:
                    goal_stats["area"][word] += 1.0

                if update_price_slot:
                    goal_stats["pricerange"][word] += 1.0

        # pick top values for each slot
        super(LearnedTracker, self).fill_goal_labels(goal_stats, hyps)
        super(LearnedTracker, self).fill_joint_goals(hyps)

        self.hyps = hyps
        return self.hyps
