import copy
import numpy as np
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity


# First iteration is to simply cosine similarity user uterrances with a updated knowledge base
# Second iteration is to try and create a BERT classifier using bert encoded user dialogs as features and their slu slots as labels

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

        if best_asr_hyp:
            encoded_asr_hyp = BertTracker.bc.encode([best_asr_hyp])

            cosine_sim_arr = cosine_similarity(encoded_asr_hyp, self.encoded_kb)
            slot_index: int = np.argmax(cosine_sim_arr)

            if cosine_sim_arr[0][
                slot_index] >= 0.80:  # if cosine sim between utterance and kb is bellow 0.75 it's probably irrelevant

                possible_slots = self.knowledge_base[slot_index]

                food, area, price = possible_slots.split(',')

                if food:
                    hyps["goal-labels"]["food"] = {
                        food: 1.0
                    }

                if area:
                    hyps["goal-labels"]["area"] = {
                        area: 1.0
                    }

                if price:
                    hyps["goal-labels"]["pricerange"] = {
                        price: 1.0
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

        food_area = [food + "," + area + "," for food in food_options for area in area_options]
        food_price = [food + ",," + price for food in food_options for price in pricerange_options]
        area_price = ["," + area + "," + price for area in area_options for price in pricerange_options]

        food_area_price = [f_a + price for f_a in food_area for price in pricerange_options]

        knowledge_base = [food + ",," for food in food_options] + ["," + area + "," for area in area_options] + \
                         [",," + price for price in pricerange_options] + \
                         food_area + food_price + area_price + food_area_price

        return knowledge_base, BertTracker.bc.encode(knowledge_base)
