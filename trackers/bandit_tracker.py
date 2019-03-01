import copy
from collections import defaultdict
from bert_serving.client import BertClient
from trackers.abstract_tracker import AbstractTracker
from contextualbandits.online import LinUCB


class BanditTracker(AbstractTracker):

    def __init__(self, ontology):
        super(BanditTracker, self).__init__(ontology)
        self.bc = BertClient(check_version=False, check_length=False)

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
        return "N/A"
