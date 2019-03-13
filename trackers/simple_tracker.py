import os
import copy
from collections import defaultdict
from trackers.abstract_tracker import AbstractTracker


class SimpleTracker(AbstractTracker):

    def __init__(self, ontology):
        """
        Initializes an instance of this class
        :param ontology: JSON object containing the ontology of the task
        """
        super(SimpleTracker, self).__init__(ontology)

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

        # Obtain the ontology information
        pricerange_options = self.ontology["informable"]["pricerange"]
        food_options = self.ontology["informable"]["food"]
        area_options = self.ontology["informable"]["area"]

        # SIMPLE Matching
        # Iterate through all the words in ontology
        # If the word is present in the user utterance update that slot with the word
        # May fail if a word in the ontology partially matches a substring of the user utterance
        for price_opt in pricerange_options:
            if price_opt in best_asr_hyp:
                goal_stats["pricerange"][price_opt] += 1.0
                # hyps["goal-labels"]["pricerange"] = {
                #     price_opt: 1.0
                # }
                # break

        for food_opt in food_options:
            if food_opt in best_asr_hyp:
                goal_stats["food"][food_opt] += 1.0
                # hyps["goal-labels"]["food"] = {
                #     food_opt: 1.0
                # }
                # break

        for area_opt in area_options:
            if area_opt in best_asr_hyp:
                goal_stats["area"][area_opt] += 1.0
                # hyps["goal-labels"]["area"] = {
                #     area_opt: 1.0
                # }
                # break

        super(SimpleTracker, self).fill_goal_labels(goal_stats, hyps)
        super(SimpleTracker, self).fill_joint_goals(hyps)

        self.hyps = hyps
        return self.hyps
