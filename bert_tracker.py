import copy
from bert_serving.client import BertClient


class BertTracker:

    def __init__(self, ontology):
        self.reset()
        self.ontology = ontology

    def reset(self):
        """Resets hypothesis dictionary to empty value"""

        self.hyps = {"goal-labels": {}, "goal-labels-joint": [], "requested-slots": {}, "method-label": {}}

    def addTurn(self, turn):
        """ Adds a turn to the tracker """

        hyps = copy.deepcopy(self.hyps)

        best_asr_hyp = turn['input']["live"]['asr-hyps'][0]["asr-hyp"]

        pricerange_options = self.ontology["informable"]["pricerange"]
        food_options = self.ontology["informable"]["food"]
        area_options = self.ontology["informable"]["area"]

        bc = BertClient(check_version=False)

        print(bc.encode([best_asr_hyp]))

        return "N/A"
