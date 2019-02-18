import copy


class CustomTracker:

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

        for price_opt in pricerange_options:
            if price_opt in best_asr_hyp:
                hyps["goal-labels"]["pricerange"] = {
                    price_opt: 1.0
                }
                break

        for food_opt in food_options:
            if food_opt in best_asr_hyp:
                hyps["goal-labels"]["food"] = {
                    food_opt: 1.0
                }
                break

        for area_opt in area_options:
            if area_opt in best_asr_hyp:
                hyps["goal-labels"]["area"] = {
                    area_opt: 1.0
                }
                break

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

        # print("{}\n{}\n".format(best_asr_hyp, self.hyps["goal-labels"]))
        self.hyps = hyps
        return self.hyps
