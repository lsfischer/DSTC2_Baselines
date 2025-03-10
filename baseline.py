import dataset_walker
import argparse, json, time, copy
from collections import defaultdict
from trackers.bert_tracker import BertTracker
from trackers.custom_tracker import CustomTracker
from trackers.bandit_tracker import BanditTracker
from trackers.learned_tracker import LearnedTracker
from trackers.bandit_tracker_tf import BanditTrackerTF
from trackers.simple_tracker import SimpleTracker


def labels(user_act, mact):
    # get context for "this" in inform(=dontcare)
    # get context for affirm and negate

    this_slot = None

    confirm_slots = {"explicit": [], "implicit": []}

    for act in mact:

        if act["act"] == "request":
            this_slot = act["slots"][0][1]

        elif act["act"] == "select":
            this_slot = act["slots"][0][0]

        elif act["act"] == "impl-conf":
            confirm_slots["implicit"] += act["slots"]

        elif act["act"] == "expl-conf":
            confirm_slots["explicit"] += act["slots"]
            this_slot = act["slots"][0][0]

    # goal_labels
    informed_goals = {}
    denied_goals = defaultdict(list)

    for act in user_act:

        act_slots = act["slots"]
        slot = None
        value = None

        if len(act_slots) > 0:
            assert len(act_slots) == 1

            if act_slots[0][0] == "this":
                slot = this_slot
            else:
                slot = act_slots[0][0]

            value = act_slots[0][1]

        if act["act"] == "inform" and slot is not None:
            informed_goals[slot] = value

        elif act["act"] == "deny" and slot is not None:
            denied_goals[slot].append(value)

        elif act["act"] == "negate":
            slot_values = confirm_slots["implicit"] + confirm_slots["explicit"]
            if len(slot_values) > 1:
                # print "Warning: negating multiple slots- it's not clear what to do."
                pass
            else:
                for slot, value in slot_values:
                    denied_goals[slot].append(value)

        elif act["act"] == "affirm":
            slot_values = confirm_slots["explicit"]
            if len(slot_values) > 1:
                # print "Warning: affirming multiple slots- it's not clear what to do."
                pass
            else:
                for slot, value in confirm_slots["explicit"]:
                    informed_goals[slot] = (value)

    # requested slots
    requested_slots = []
    for act in user_act:
        if act["act"] == "request":
            for _, requested_slot in act["slots"]:
                requested_slots.append(requested_slot)
    # method
    method = "none"
    act_types = [act["act"] for act in user_act]
    mact_types = [act["act"] for act in mact]

    if "reqalts" in act_types:
        method = "byalternatives"
    elif "bye" in act_types:
        method = "finished"
    elif "inform" in act_types:
        method = "byconstraints"
        for act in [uact for uact in user_act if uact["act"] == "inform"]:
            slots = [slot for slot, _ in act["slots"]]
            if "name" in slots:
                method = "byname"

    return informed_goals, denied_goals, requested_slots, method


def Uacts(turn):
    """ return merged slu-hyps, replacing "this" with the correct slot """

    mact = []

    if "dialog-acts" in turn["output"]:
        mact = turn["output"]["dialog-acts"]

    this_slot = None

    for act in mact:
        # get the requested slot by the user and save it in this_slot
        if act["act"] == "request":
            this_slot = act["slots"][0][1]

    this_output = []

    for slu_hyp in turn['input']["live"]['slu-hyps']:
        score = slu_hyp['score']
        this_slu_hyp = slu_hyp['slu-hyp']
        these_hyps = []

        for hyp in this_slu_hyp:

            for i in range(len(hyp["slots"])):
                slot, _ = hyp["slots"][i]

                if slot == "this":
                    hyp["slots"][i][0] = this_slot

            these_hyps.append(hyp)

        this_output.append((score, these_hyps))

    this_output.sort(key=lambda x: x[0], reverse=True)

    return this_output


class Tracker(object):

    def __init__(self):
        self.reset()

    def addTurn(self, turn):

        """ Adds a turn to the tracker """

        hyps = copy.deepcopy(self.hyps)

        if "dialog-acts" in turn["output"]:
            mact = turn["output"]["dialog-acts"]
        else:
            mact = []

        # clear requested-slots that have been informed
        for act in mact:
            if act["act"] == "inform":
                for slot, value in act["slots"]:
                    if slot in hyps["requested-slots"]:
                        hyps["requested-slots"][slot] = 0.0

        # Gets a list of tuples containing all slu hypothesis for that turn and the score for how probable they are
        slu_hyps = Uacts(turn)

        requested_slot_stats = defaultdict(float)
        method_stats = defaultdict(float)
        goal_stats = defaultdict(lambda: defaultdict(float))
        prev_method = "none"

        if len(hyps["method-label"].keys()) > 0:
            prev_method = list(hyps["method-label"].keys())[0]

        for score, uact in slu_hyps:

            informed_goals, denied_goals, requested, method = labels(uact, mact)

            # requesteds
            for slot in requested:
                requested_slot_stats[slot] += score
            if method == "none":
                method = prev_method
            method_stats[method] += score

            # goal_labels
            for slot in informed_goals:
                value = informed_goals[slot]
                goal_stats[slot][value] += score

        # pick top values for each slot
        for slot in goal_stats:
            curr_score = 0.0
            if (slot in hyps["goal-labels"]):
                curr_score = list(hyps["goal-labels"][slot].values())[0]
            for value in goal_stats[slot]:
                score = goal_stats[slot][value]
                if score >= curr_score:
                    hyps["goal-labels"][slot] = {
                        value: clip(score)
                    }
                    curr_score = score

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

        for slot in requested_slot_stats:
            hyps["requested-slots"][slot] = clip(requested_slot_stats[slot])

        # normalise method_stats    
        hyps["method-label"] = normalise_dict(method_stats)
        self.hyps = hyps
        return self.hyps

    def reset(self):
        """Resets hypothesis dictionary to empty value"""

        self.hyps = {"goal-labels": {}, "goal-labels-joint": [], "requested-slots": {}, "method-label": {}}


class FocusTracker(object):
    # only track goals, don't do requested slots and method
    def __init__(self):
        self.reset()

    def addTurn(self, turn):
        hyps = copy.deepcopy(self.hyps)
        if "dialog-acts" in turn["output"]:
            mact = turn["output"]["dialog-acts"]
        else:
            mact = []
        slu_hyps = Uacts(turn)

        this_u = defaultdict(lambda: defaultdict(float))
        method_stats = defaultdict(float)
        requested_slot_stats = defaultdict(float)
        for score, uact in slu_hyps:
            informed_goals, denied_goals, requested, method = labels(uact, mact)
            method_stats[method] += score
            for slot in requested:
                requested_slot_stats[slot] += score
            # goal_labels
            for slot in informed_goals:
                this_u[slot][informed_goals[slot]] += score

        for slot in this_u.keys() + hyps["goal-labels"].keys():
            q = max(0.0, 1.0 - sum(
                [this_u[slot][value] for value in this_u[slot]]))  # clipping at zero because rounding errors
            if slot not in hyps["goal-labels"]:
                hyps["goal-labels"][slot] = {}

            for value in hyps["goal-labels"][slot]:
                hyps["goal-labels"][slot][value] *= q
            prev_values = hyps["goal-labels"][slot].keys()
            for value in this_u[slot]:
                if value in prev_values:
                    hyps["goal-labels"][slot][value] += this_u[slot][value]
                else:
                    hyps["goal-labels"][slot][value] = this_u[slot][value]

            hyps["goal-labels"][slot] = normalise_dict(hyps["goal-labels"][slot])

        # method node, in 'focus' manner:
        q = min(1.0, max(0.0, method_stats["none"]))
        method_label = hyps["method-label"]
        for method in method_label:
            if method != "none":
                method_label[method] *= q
        for method in method_stats:
            if method == "none":
                continue
            if method not in method_label:
                method_label[method] = 0.0
            method_label[method] += method_stats[method]

        if "none" not in method_label:
            method_label["none"] = max(0.0, 1.0 - sum(method_label.values()))

        hyps["method-label"] = normalise_dict(method_label)

        # requested slots
        informed_slots = []
        for act in mact:
            if act["act"] == "inform":
                for slot, value in act["slots"]:
                    informed_slots.append(slot)

        for slot in (requested_slot_stats.keys() + hyps["requested-slots"].keys()):
            p = requested_slot_stats[slot]
            prev_p = 0.0
            if slot in hyps["requested-slots"]:
                prev_p = hyps["requested-slots"][slot]
            x = 1.0 - float(slot in informed_slots)
            new_p = x * prev_p + p
            hyps["requested-slots"][slot] = clip(new_p)

        self.hyps = hyps
        return self.hyps

    def reset(self):
        self.hyps = {"goal-labels": {}, "method-label": {}, "requested-slots": {}}


def clip(x):
    if x > 1:
        return 1
    if x < 0:
        return 0
    return x


def normalise_dict(x):
    x_items = x.items()
    total_p = sum([p for k, p in x_items])
    if total_p > 1.0:
        x_items = [(k, p / total_p) for k, p in x_items]
    return dict(x_items)


def main():
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True, metavar='PATH',
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile', dest='trackfile', action='store', required=True, metavar='JSON_FILE',
                        help='File to write with tracker output')
    parser.add_argument('--focus', dest='focus', action='store', nargs='?', default="False", const="True",
                        help='Use focus node tracker')
    parser.add_argument('--config', dest='config', action='store', required=True, metavar='TRUE/FALSE',
                        help='The path of the config folder containing the .flist files')
    parser.add_argument('--tracker', dest='tracker', action='store', nargs='?', default="LearnedTracker",
                        help='Tracker to use')
    parser.add_argument('--ontology', dest='ontology', action='store', metavar='JSON_FILE', required=True,
                        help='The ontology to use')

    args = parser.parse_args()

    # Opens data set file and stores it in dataset object
    dataset = dataset_walker.dataset_walker(args.dataset, dataroot=args.dataroot, config_folder=args.config)

    # Opens track file
    track_file = open(args.trackfile, "w")

    track = {"sessions": [], "dataset": args.dataset}

    start_time = time.time()

    # Choosing what kind of tracker to use
    if args.tracker.lower() == "tracker":
        tracker = Tracker()

    elif args.tracker.lower() == "focustracker":
        tracker = FocusTracker()

    elif args.tracker.lower() == "customtracker":
        ontology = json.load(open(args.ontology))
        tracker = CustomTracker(ontology)

    elif args.tracker.lower() == "berttracker":
        ontology = json.load(open(args.ontology))
        tracker = BertTracker(ontology)

    elif args.tracker.lower() == "learnedtracker":
        ontology = json.load(open(args.ontology))
        tracker = LearnedTracker(ontology)

    elif args.tracker.lower() == "bandittracker":
        ontology = json.load(open(args.ontology))
        tracker = BanditTracker(ontology)

    elif args.tracker.lower() == "bandittrackertf":
        ontology = json.load(open(args.ontology))
        tracker = BanditTrackerTF(ontology)

    elif args.tracker.lower() == "simpletracker":
        ontology = json.load(open(args.ontology))
        tracker = SimpleTracker(ontology)

        # Iterates over every call in the dataset
    for call in dataset:
        this_session = {"session-id": call.log["session-id"], "turns": []}
        tracker.reset()

        # Iterates over every turn in a call
        for turn, _ in call:
            # Adds the turn to the tracker
            tracker_turn = tracker.addTurn(turn, call.log["session-id"])

            this_session["turns"].append(tracker_turn)

        track["sessions"].append(this_session)

    end_time = time.time()
    elapsed_time = end_time - start_time

    track["wall-time"] = elapsed_time

    json.dump(track, track_file, indent=4)


if __name__ == '__main__':
    main()
