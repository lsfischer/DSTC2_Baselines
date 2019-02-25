import pickle
import string
import copy
import numpy as np
from bert_serving.client import BertClient
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams


class LearnedTracker:

    def __init__(self, ontology):
        self.reset()
        self.ontology = ontology
        self.bc = BertClient(check_version=False, check_length=False)
        self.food_classifier, self.area_classifier, self.price_classifier = self.train()

    def reset(self):
        """Resets hypothesis dictionary to empty value"""
        self.hyps = {"goal-labels": {}, "goal-labels-joint": [], "requested-slots": {}, "method-label": {}}

    def is_word_in_ontology(self, word, slot_type="food"):

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

        print("training area")
        best_area_c = 51.0  # self.train_aux(area_training_data)

        print("training price")
        best_price_c = 1.0  # self.train_aux(price_training_data)

        food_classifier = svm.SVC(kernel="rbf", C=best_food_c, gamma="auto")
        area_classifier = svm.SVC(kernel="rbf", C=best_area_c, gamma="auto")
        price_classifier = svm.SVC(kernel="rbf", C=best_price_c, gamma="auto")

        food_classifier.fit(np.array(food_training_data["features"]), np.array(food_training_data["labels"]))
        area_classifier.fit(np.array(area_training_data["features"]), np.array(area_training_data["labels"]))
        price_classifier.fit(np.array(price_training_data["features"]), np.array(price_training_data["labels"]))

        return food_classifier, area_classifier, price_classifier  # 131.0 51.0 1.0

    def addTurn(self, turn):
        """ Adds a turn to the tracker """

        hyps = copy.deepcopy(self.hyps)
        best_asr_hyp = turn['input']["live"]['asr-hyps'][0]["asr-hyp"]
        stop = stopwords.words('english') + list(string.punctuation)
        tknz = word_tokenize(best_asr_hyp)
        processed_hyp = [word for word in tknz if
                         word not in stop]  # + [tup[0] + " " + tup[1] for tup in ngrams(tknz, 2)]

        for idx, word in enumerate(processed_hyp):
            if word == "moderately" or word == "affordable":
                processed_hyp[idx] = "moderate"
            if word == "cheaper":
                processed_hyp[idx] = "cheap"

        if processed_hyp:

            sentence_embedding = np.array(self.bc.encode([best_asr_hyp]))[0]

            for word in processed_hyp:
                word_embedding = np.array(self.bc.encode([word]))[0]

                word_in_food_ontology = [self.is_word_in_ontology(word, slot_type="food")]
                word_in_area_ontology = [self.is_word_in_ontology(word, slot_type="area")]
                word_in_price_ontology = [self.is_word_in_ontology(word, slot_type="price")]

                food_features = np.concatenate((word_embedding, sentence_embedding, word_in_food_ontology))
                area_features = np.concatenate((word_embedding, sentence_embedding, word_in_area_ontology))
                price_features = np.concatenate((word_embedding, sentence_embedding, word_in_price_ontology))

                update_food_slot = self.food_classifier.predict([food_features])[0]
                update_area_slot = self.area_classifier.predict([area_features])[0]
                update_price_slot = self.price_classifier.predict([price_features])[0]

                if update_food_slot:
                    hyps["goal-labels"]["food"] = {
                        word: 1.0  # Change for some other value
                    }

                if update_area_slot:
                    hyps["goal-labels"]["area"] = {
                        word: 1.0  # Change for some other value
                    }

                if update_price_slot:
                    hyps["goal-labels"]["pricerange"] = {
                        word: 1.0  # Change for some other value
                    }

            informed_slots = list(hyps["goal-labels"].keys())
            for inf_slot in informed_slots:
                if len(hyps["goal-labels-joint"]) > 0:
                    hyps["goal-labels-joint"][0]["slots"][inf_slot] = list(hyps["goal-labels"][inf_slot].keys())[0]
                else:
                    obj = {
                        "slots": {
                            inf_slot: list(hyps["goal-labels"][inf_slot].keys())[0]
                        },
                        "score": 1.0
                    }
                    hyps["goal-labels-joint"].append(obj)

        self.hyps = hyps
        return self.hyps
