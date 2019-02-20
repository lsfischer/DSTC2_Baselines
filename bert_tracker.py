import copy
import numpy as np
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity


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
        encoded_asr_hyp = BertTracker.bc.encode([best_asr_hyp])

        slot_index = np.argmax(cosine_similarity(encoded_asr_hyp, self.encoded_kb))

        possible_slots = self.knowledge_base[slot_index]
        print("Utterance: {}\nSlots: {}\n".format(best_asr_hyp, possible_slots))

        return "N/A"

    def encode_ontology(self):
        """Encodes the ontology JSON using BERT"""

        ontology = copy.deepcopy(self.ontology)

        pricerange_options = ontology["informable"]["pricerange"]
        food_options = ontology["informable"]["food"]
        area_options = ontology["informable"]["area"]

        food_area = [food + "," + area + "," for food in food_options for area in area_options]
        food_price = [food + ",," + price for food in food_options for price in pricerange_options]
        area_price = ["," + area + "," + price for area in area_options for price in pricerange_options]

        food_area_price = [f_a + price for f_a in food_area for price in pricerange_options]

        knowledge_base = [food + ",," for food in food_options] + ["," + area + "," for area in area_options] + \
                         [",," + price for price in pricerange_options] + \
                         food_area + food_price + area_price + food_area_price

        return knowledge_base, BertTracker.bc.encode(knowledge_base)
