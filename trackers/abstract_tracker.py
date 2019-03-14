from abc import ABC, abstractmethod


def clip(x):
    if x > 1:
        return 1
    if x < 0:
        return 0
    return x


class AbstractTracker(ABC):

    def __init__(self, ontology):
        """Initializes future objects that extend this class"""
        self.reset()
        self.ontology = ontology

    def reset(self):
        """Resets hypothesis dictionary to empty value"""

        self.hyps = {"goal-labels": {}, "goal-labels-joint": [], "requested-slots": {}, "method-label": {}}

    def fill_goal_labels(self, goal_stats, hyps):
        """
        Fills the goal-labels slot values with information obtained from goal_stats. Only uses the top score of each slot to fill that slot
        :param goal_stats: Dictionary containing the slot values and scores
        :param hyps: The dialog state object from which to obtain the information
        """
        for slot in goal_stats:
            curr_score = 0.0

            if slot in hyps["goal-labels"]:
                curr_score = list(hyps["goal-labels"][slot].values())[0]

            for value in goal_stats[slot]:
                score = goal_stats[slot][value]

                if score >= curr_score:
                    hyps["goal-labels"][slot] = {
                        value: clip(score)
                    }
                    curr_score = score

    def fill_joint_goals(self, hyps):
        """
        Fills the goal-labels-joint object with information obtained from previous updated slots, present in hyps
        :param hyps: The dialog state object from which to obtain the information
        """

        # joint estimate is the above selection, with geometric mean score
        goal_joint_label = {"slots": {}, "scores": []}

        for slot in hyps["goal-labels"]:
            (value, score), = hyps["goal-labels"][slot].items()

            if score < 0.5:
                # then None is more likely
                continue

            goal_joint_label["scores"].append(score)
            goal_joint_label["slots"][slot] = value

        if len(goal_joint_label["slots"]) > 0:
            geom_mean = 1.0

            for score in goal_joint_label["scores"]:
                geom_mean *= score

            geom_mean = geom_mean ** (1.0 / len(goal_joint_label["scores"]))
            goal_joint_label["score"] = clip(geom_mean)

            del goal_joint_label["scores"]

            hyps["goal-labels-joint"] = [goal_joint_label]

    @abstractmethod
    def addTurn(self, turn, session_id):
        """ Adds a turn to the tracker """
        raise NotImplementedError
