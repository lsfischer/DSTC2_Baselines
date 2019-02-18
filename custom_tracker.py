import copy


def remove_informed_slots(mact, requested_slots):
    """ clear requested-slots that have been informed """

    for act in mact:
        if act["act"] == "inform":
            for slot, value in act["slots"]:
                if slot in requested_slots.keys():
                    requested_slots[slot] = 0.0


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

        if "dialog-acts" in turn["output"]:
            sys_acts = turn["output"]["dialog-acts"]
        else:
            sys_acts = []

        remove_informed_slots(sys_acts, hyps["requested-slots"])

        best_asr_hyp = turn['input']["live"]['asr-hyps'][0]

        print(self.ontology["pricerange"])

        return "N/A"
