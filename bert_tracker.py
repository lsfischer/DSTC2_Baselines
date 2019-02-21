import copy
import numpy as np
import math
import string
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams


class BertTracker:
    bc = BertClient(check_version=False)

    def __init__(self, ontology):
        self.reset()
        self.ontology = ontology
        self.knowledge_base, self.encoded_kb = self.encode_ontology()

    def reset(self):
        """Resets hypothesis dictionary to empty value"""

        self.hyps = {"goal-labels": {}, "goal-labels-joint": [], "requested-slots": {}, "method-label": {}}

    def addTurn(self, turn):
        """ Adds a turn to the tracker """

        hyps = copy.deepcopy(self.hyps)
        best_asr_hyp = turn['input']["live"]['asr-hyps'][0]["asr-hyp"]
        stop = stopwords.words('english') + list(string.punctuation)
        tknz = word_tokenize(best_asr_hyp)
        processed_hyp = [word for word in tknz if word not in stop] + [tup[0] + " " + tup[1] for tup in ngrams(tknz, 2)]

        for idx, word in enumerate(processed_hyp):
            if word == "moderately" or word == "affordable":
                processed_hyp[idx] = "moderate"
            if word == "cheaper":
                processed_hyp[idx] = "cheap"

        if processed_hyp:

            pricerange_options = self.ontology["informable"]["pricerange"]
            food_options = self.ontology["informable"]["food"]
            area_options = self.ontology["informable"]["area"]

            slots_to_fill = []
            for hyp_word in processed_hyp:

                if hyp_word in food_options:
                    slots_to_fill.append((hyp_word, "food"))

                if hyp_word in area_options:
                    slots_to_fill.append((hyp_word, "area"))

                if hyp_word in pricerange_options:
                    slots_to_fill.append((hyp_word, "pricerange"))

            if len(slots_to_fill) != 0:
                # If a simple  matching was able to find some results
                food_slots = [tup[0] for tup in slots_to_fill if tup[1] == "food"]
                area_slots = [tup[0] for tup in slots_to_fill if tup[1] == "area"]
                pricerange_slots = [tup[0] for tup in slots_to_fill if tup[1] == "pricerange"]

                if food_slots:
                    hyps["goal-labels"]["food"] = {
                        food_slots[-1]: 1.0
                    }

                if area_slots:
                    hyps["goal-labels"]["area"] = {
                        area_slots[-1]: 1.0
                    }

                if pricerange_slots:
                    hyps["goal-labels"]["pricerange"] = {
                        pricerange_slots[-1]: 1.0
                    }

            else:
                # When simple matching does not find anything, try using bert to infer different lexical forms
                encoded_hyp = np.array(BertTracker.bc.encode(processed_hyp))

                cosine_sim = cosine_similarity(encoded_hyp, self.encoded_kb)

                for idx, sub_arr in enumerate(cosine_sim):
                    argmax_index = np.argmax(sub_arr)

                    if sub_arr[argmax_index] >= 0.97:
                        # assuming that if it's lower than 0.97 then it's probably a mistake
                        kb_word = self.knowledge_base[argmax_index]
                        print(f"BERT: Word in query: {processed_hyp[idx]} \t matched with {kb_word}")

                        if kb_word in food_options:
                            hyps["goal-labels"]["food"] = {
                                kb_word: 1.0
                            }

                        if kb_word in area_options:
                            hyps["goal-labels"]["area"] = {
                                kb_word: 1.0
                            }

                        if kb_word in pricerange_options:
                            hyps["goal-labels"]["pricerange"] = {
                                kb_word: 1.0
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

    def encode_ontology(self):
        """Encodes the ontology JSON using BERT"""

        ontology = copy.deepcopy(self.ontology)

        pricerange_options = ontology["informable"]["pricerange"]
        food_options = ontology["informable"]["food"]
        area_options = ontology["informable"]["area"]

        # food_area = [food + "," + area + "," for food in food_options for area in area_options]
        # food_price = [food + ",," + price for food in food_options for price in pricerange_options]
        # area_price = ["," + area + "," + price for area in area_options for price in pricerange_options]

        # food_area_price = [f_a + price for f_a in food_area for price in pricerange_options]
        #
        # knowledge_base = [food + ",," for food in food_options] + ["," + area + "," for area in area_options] + \
        #                  [",," + price for price in pricerange_options] + \
        #                  food_area + food_price + area_price + food_area_price

        knowledge_base = food_options + area_options + pricerange_options

        return knowledge_base, np.array(BertTracker.bc.encode(knowledge_base))
