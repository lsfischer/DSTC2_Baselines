import os
import copy
import json
from collections import defaultdict
from trackers.abstract_tracker import AbstractTracker


class SimpleTracker(AbstractTracker):

    def __init__(self, ontology):
        """
        Initializes an instance of this class
        :param ontology: JSON object containing the ontology of the task
        """
        super(SimpleTracker, self).__init__(ontology)
        self.dataset_path = "/nas/Datasets/dstc/dstc2/traindev/data/"

    def addTurn(self, turn, session_id):
        """
        Adds a turn to this tracker
        :param turn: The turn to process and add
        :return: A hypothesis of the current state of the dialog
        """

        hyps = copy.deepcopy(self.hyps)

        goal_stats = defaultdict(lambda: defaultdict(float))
        label_file_path = None

        for root, _, _ in os.walk(self.dataset_path):
            if session_id in root:
                label_file_path = f"{root}/label.json"
                break

        turn_idx = turn["turn-index"]

        # Open the label.json file
        label_file = json.load(open(label_file_path))

        # Getting the user utterance from the label file
        user_utterance = label_file["turns"][turn_idx]["transcription"]

        # Obtain the ontology information
        pricerange_options = self.ontology["informable"]["pricerange"]
        food_options = self.ontology["informable"]["food"]
        area_options = self.ontology["informable"]["area"]

        # SIMPLE Matching
        # Iterate through all the words in ontology
        # If the word is present in the user utterance update that slot with the word
        # May fail if a word in the ontology partially matches a substring of the user utterance
        for price_opt in pricerange_options:
            if price_opt in user_utterance:
                goal_stats["pricerange"][price_opt] += 1.0

        for food_opt in food_options:
            if food_opt in user_utterance:
                goal_stats["food"][food_opt] += 1.0

        for area_opt in area_options:
            if area_opt in user_utterance:
                goal_stats["area"][area_opt] += 1.0

        super(SimpleTracker, self).fill_goal_labels(goal_stats, hyps)
        super(SimpleTracker, self).fill_joint_goals(hyps)

        self.hyps = hyps
        return self.hyps
