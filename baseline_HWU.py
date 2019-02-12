import argparse, json, time, copy, math
import dataset_walker
from collections import namedtuple
from argparse import RawTextHelpFormatter

Ontology = {}
SLU_Hypo = namedtuple("SLU_Hypo", ["act", "slot", "value"])


def load_ontology(file):
    global Ontology
    json_data = open(file)
    Ontology = json.load(json_data)


def what_slot(mact):
    # get context for "this" in inform(=dontcare)
    this_slot = None

    for act in mact:
        if act["act"] == "request":
            this_slot = act["slots"][0][1]
        elif act["act"] == "select":
            this_slot = act["slots"][0][0]
        elif act["act"] == "expl-conf":
            this_slot = act["slots"][0][0]

    return this_slot


def pause():
    input('press enter to continue ...')


def Uacts(turn):
    # return merged slu-hyps, replacing "this" with the correct slot
    mact = []
    if "dialog-acts" in turn["output"]:
        mact = turn["output"]["dialog-acts"]
    this_slot = None
    for act in mact:
        if act["act"] == "request":
            this_slot = act["slots"][0][1]
    this_output = []
    for slu_hyp in turn['input']["live"]['slu-hyps']:
        score = slu_hyp['score']
        this_slu_hyp = slu_hyp['slu-hyp']
        these_hyps = []
        for hyp in this_slu_hyp:
            for i in range(len(hyp["slots"])):
                slot, value = hyp["slots"][i]
                if slot == "this":
                    hyp["slots"][i][0] = this_slot
                elif slot == None and value == "dontcare":
                    hyp["slots"][i][0] = what_slot(mact)
            these_hyps.append(hyp)
        this_output.append((score, these_hyps))
    this_output.sort(key=lambda x: x[0], reverse=True)
    return this_output


def is_informable(slot):
    global Ontology
    return slot in Ontology["informable"].keys()


def get_slot_value(dact):
    a = dact["act"]
    s = None;
    v = None;
    if dact["slots"]:
        if len(dact["slots"][0]) >= 1:
            s = dact["slots"][0][0]
        if len(dact["slots"][0]) == 2:
            v = dact["slots"][0][1]
    return a, s, v


def remove_dup(mylist):
    newlist = []
    for elm in mylist:
        if not elm in newlist:
            newlist.append(elm)
    return newlist


def rule_affirm_negate(uact, score, mact, marg_hyps):
    # rule to convert compact affirm/negate user acts

    found_affirm = False
    found_inform = False
    found_negate = False
    updated = False
    for dact in uact:
        if dact["act"] == 'inform':
            found_inform = True
        if dact["act"] == 'affirm':
            found_affirm = True
        elif dact["act"] == 'negate':
            found_negate = True

    if found_inform and (found_affirm or found_negate):
        # if there are 'inform' following 'affirm' or 'negate', only consider the informed slot values
        for dact in uact:
            a, s, v = get_slot_value(dact)
            if a != 'inform' or not is_informable(s): continue
            hyp = SLU_Hypo(act=a, slot=s, value=v)
            if hyp in marg_hyps.keys():
                marg_hyps[hyp] = clip(marg_hyps[hyp] + score)
            else:
                marg_hyps[hyp] = score

    elif found_affirm:
        # else treat all the slot values in 'impl-conf/expl-conf' system actions as being affirmed
        for dact in mact:
            a, s, v = get_slot_value(dact)
            if a in ['expl-conf', 'impl-conf'] and is_informable(s):
                hyp = SLU_Hypo(act='inform', slot=s, value=v)
                if hyp in marg_hyps.keys():
                    marg_hyps[hyp] = clip(marg_hyps[hyp] + score)
                else:
                    marg_hyps[hyp] = score

    elif found_negate:
        # else treat all the slot values in 'impl-conf/expl-conf' system actions as being negated
        for dact in mact:
            a, s, v = get_slot_value(dact)
            if a in ['expl-conf', 'impl-conf'] and is_informable(s):
                hyp = SLU_Hypo(act='deny', slot=s, value=v)
                if hyp in marg_hyps.keys():
                    marg_hyps[hyp] = clip(marg_hyps[hyp] + score)
                else:
                    marg_hyps[hyp] = score

    updated = found_affirm or found_negate
    return updated


def rule_inform_deny(marg_hyps, goals):
    # better handle goal change
    for slot in goals.keys():
        for value in goals[slot].keys():
            conflict = 0.0
            confirm = 0.0
            for hyp, score in marg_hyps.items():
                if hyp.slot == slot:
                    if hyp.value == value:
                        if hyp.act == 'inform':
                            confirm += score
                        elif hyp.act == 'deny':
                            conflict += score
                    elif hyp.act == 'inform':
                        conflict += score

            if confirm + conflict > 1.0:
                confirm = confirm / (confirm + conflict)
                conflict = conflict / (confirm + conflict)

            goals[slot][value] = clip(1.0 - (1 - goals[slot][value]) * (1.0 - confirm))
            goals[slot][value] = clip(goals[slot][value] * (1.0 - conflict))

    for hyp, score in marg_hyps.items():
        if hyp.act == 'inform':
            if not is_informable(hyp.slot): continue
            if not hyp.slot in goals.keys():
                goals[hyp.slot] = {}
                goals[hyp.slot][hyp.value] = score
            elif not hyp.value in goals[hyp.slot].keys():
                goals[hyp.slot][hyp.value] = score


def rule_impl_affirm(act, marg_hyps, goals):
    # if system implicitly confirms ('impl-conf') some information,
    # and no explicit (stand-alone) affirm/negate or any confliction is observed, 
    # boost the probabolity for the presented information to 1, see (Wang & Lemon, SigDial 2013)
    # this rule is only used in the 'original' version
    a, s, v = get_slot_value(act)
    if not is_informable(s): return
    found_conflict = False
    for hyp in marg_hyps:
        if hyp.act in ['inform', 'deny'] and hyp.slot == s:
            found_conflict = True
            break
    if not found_conflict:
        if not s in goals.keys():
            goals[s] = {}
        goals[s][v] = 1.0


def rule_inform(hyp, score, goals):
    # update rule for 'inform' user acts
    if not is_informable(hyp.slot): return
    if not hyp.slot in goals.keys():
        goals[hyp.slot] = {}
        goals[hyp.slot][hyp.value] = score
    else:
        if not hyp.value in goals[hyp.slot].keys():
            goals[hyp.slot][hyp.value] = score
        else:
            goals[hyp.slot][hyp.value] = clip(1.0 - (1.0 - goals[hyp.slot][hyp.value]) * (1.0 - score))


def rule_deny(hyp, score, goals):
    # update rule for 'inform' user acts
    if not is_informable(hyp.slot): return
    if not hyp.slot in goals.keys():
        return
    else:
        if not hyp.value in goals[hyp.slot].keys():
            return
        else:
            goals[hyp.slot][hyp.value] = clip(goals[hyp.slot][hyp.value] * (1.0 - score))


def rule_canthelp(act, blocked_single, blocked_joint):
    sv = act["slots"]
    if len(sv) == 1:
        s = sv[0][0]
        v = sv[0][1]
        if not (s, v) in blocked_single:
            blocked_single.append((s, v))
    else:
        c = {}
        for t in sv:
            s = t[0]
            v = t[1]
            c[s] = v
        if not c in blocked_joint:
            blocked_joint.append(c)


def rule_canthelp_missing_slot_value(act, goals, requested_slots):
    sv = act["slots"]
    if len(sv) != 2: return
    name = ''
    slot = ''
    for t in sv:
        if t[0] == 'name':
            name = t[1]
        elif t[0] == 'slot':
            slot = t[1]
    if 'name' in goals.keys() and name in goals["name"].keys():
        if slot in requested_slots.keys():
            requested_slots[slot] *= clip(1.0 - goals["name"][name])


def rule_method(slu_hyps, method):
    # exactly the same strategy as the 'focus' baseline
    def is_act(uact, type, slot=None):
        for dact in uact:
            a, s, v = get_slot_value(dact)
            if a == type:
                if slot == None:
                    return True
                elif s == slot:
                    return True

        return False

    observed_method = {"byconstraints": 0.0, "byname": 0.0, "byalternatives": 0.0, "finished": 0.0}

    for score, uact in slu_hyps:
        if is_act(uact, 'reqalts'):
            observed_method["byalternatives"] = clip(observed_method["byalternatives"] + score)
        elif is_act(uact, 'bye'):
            observed_method["finished"] = clip(observed_method["finished"] + score)
        elif is_act(uact, 'inform', 'name'):
            observed_method["byname"] = clip(observed_method["byname"] + score)
        elif is_act(uact, 'inform'):
            observed_method["byconstraints"] = clip(observed_method["byconstraints"] + score)

    marg_prob = sum([p for k, p in observed_method.items()])
    method = dict([(k, p * clip(1.0 - marg_prob)) for k, p in method.items()])
    for m in get_nonzero(observed_method):
        if not m in method.keys():
            method[m] = observed_method[m]
        else:
            method[m] = observed_method[m] + method[m]

    return method


def rule_requested_slot(marg_hyps, requested_slots):
    # rule for 'request' user act, similar to 'inform'
    for hyp, score in marg_hyps.items():
        if hyp.act != 'request': continue
        if hyp.value not in requested_slots.keys():
            requested_slots[hyp.value] = score
        else:
            requested_slots[hyp.value] = clip(1.0 - (1.0 - requested_slots[hyp.value]) * (1.0 - score))


def remove_informed_slots(mact, requested_slots):
    # clear requested-slots that have been informed
    for act in mact:
        if act["act"] == "inform":
            for slot, value in act["slots"]:
                if slot in requested_slots.keys():
                    requested_slots[slot] = 0.0


def split_merge(slu_hyps, mact):
    # return marginal slu hypotheses
    marg_hyps = {}
    for score, uact in slu_hyps:
        uact = remove_dup(uact)
        updated = rule_affirm_negate(uact, score, mact, marg_hyps)
        if updated:
            continue

        for dact in uact:
            a, s, v = get_slot_value(dact)
            if s == None and a == 'inform' and v == 'dontcare':
                s = what_slot(mact)
            hyp = SLU_Hypo(act=a, slot=s, value=v)
            if hyp in marg_hyps.keys():
                marg_hyps[hyp] = clip(marg_hyps[hyp] + score)
            else:
                marg_hyps[hyp] = score

    return marg_hyps


def betafit(stats):
    # naive maximum likelihood approximation of beta distribution parameters
    # though very rough and inaccurate
    Gx = 1.0
    G1_x = 1.0
    N = len(stats)
    for x in stats:
        if x >= 1.0: x = 0.999999
        Gx *= x ** (1.0 / N)
        G1_x = (1 - x) ** (1.0 / N)

    if is_zero(1.0 - Gx - G1_x):  # division by zero
        return 1.0, 1.0

    a = 0.5 + Gx / (2.0 * (1.0 - Gx - G1_x))
    b = 0.5 + G1_x / (2.0 * (1.0 - Gx - G1_x))

    return a, b


def adjust_noise(slu_hyps, stats):
    for i in range(0, len(slu_hyps)):
        if i >= len(slu_hyps): break
        score, uact = slu_hyps[i]
        if not uact:
            del slu_hyps[i]

    f = sum([score for score, _ in slu_hyps])

    m = 1.0

    if slu_hyps or stats:
        stats.append(f)
    if stats:
        a, b = betafit(stats)
        m = a / (a + b)

    slu_hyps = [(score / m, hyp) for score, hyp in slu_hyps]

    f = sum([score for score, _ in slu_hyps])
    if f > 1.0:
        slu_hyps = [(score / f, hyp) for score, hyp in slu_hyps]

    return slu_hyps


def append_one_slot(joint_hyps, slot, values_hyps, blocked_joint):
    def is_blocked(hyp, block_list):
        for h in block_list:
            blocked = True
            for s, v in h.items():
                if not (s, v) in hyp.items():
                    blocked = False
                    break
            if blocked: return True
        return False

    bound = len(joint_hyps)
    for i in range(0, bound):
        for value, score in values_hyps.items():
            h = copy.deepcopy(joint_hyps[i])
            if is_zero(score): continue
            h["slots"][slot] = value
            if is_blocked(h, blocked_joint): continue
            h["score"] *= score
            joint_hyps.append(h)

    null_prob = clip(1 - sum([score for _, score in values_hyps.items()]))

    if is_zero(null_prob):
        del joint_hyps[0:bound]
    else:
        for i in range(0, bound):
            joint_hyps[i]["score"] *= null_prob

    return joint_hyps


def get_joint_hyps(slot_hyps, blocked_joint):
    joint_hyps = [{"slots": {}, "score": 1.0}]
    n_slot = 0;
    for slot, value_hyps in slot_hyps.items():
        if is_informable(slot):
            append_one_slot(joint_hyps, slot, value_hyps, blocked_joint)
            n_slot += 1

    if joint_hyps:
        sorted_list = sorted(joint_hyps, key=lambda x: x["score"], reverse=True)
        if sorted_list[0]["slots"]:
            return sorted_list
        else:
            del sorted_list[0]
            return sorted_list

    return []


class HWU_Tracker(object):
    def __init__(self):
        self.reset()

    def addTurn(self, turn, original=False):

        hyps = copy.deepcopy(self.hyps)
        if "dialog-acts" in turn["output"]:
            mact = turn["output"]["dialog-acts"]
        else:
            mact = []

        remove_informed_slots(mact, hyps["requested-slots"])

        slu_hyps = Uacts(turn)

        if not original:
            slu_hyps = adjust_noise(slu_hyps, self.stats)

        marg_hyps = split_merge(slu_hyps, mact)

        if not original:
            rule_inform_deny(marg_hyps, hyps["goal-labels"])
        else:
            for h, score in marg_hyps.items():
                if h.act == 'inform':
                    rule_inform(h, score, hyps["goal-labels"])
                elif h.act == 'deny':
                    rule_deny(h, score, hyps["goal-labels"])

        for (s, v) in self.blocked_single:
            if s in hyps["goal-labels"].keys() and v in hyps["goal-labels"][s].keys():
                del hyps["goal-labels"][s][v]

        hyps["method-label"] = rule_method(slu_hyps, hyps["method-label"])
        rule_requested_slot(marg_hyps, hyps["requested-slots"])

        for act in mact:
            if act["act"] == 'canthelp':
                rule_canthelp(act, self.blocked_single, self.blocked_joint)
            elif act["act"] == 'canthelp.missing_slot_value':
                rule_canthelp_missing_slot_value(act, hyps["goal-labels"], hyps["requested-slots"])
            elif original and act["act"] == 'impl-conf':
                rule_impl_affirm(act, marg_hyps, hyps["goal-labels"])

        self.hyps = copy.deepcopy(hyps)

        for slot in hyps["goal-labels"].keys():
            hyps["goal-labels"][slot] = normalise_dict(hyps["goal-labels"][slot])

        hyps["goal-labels-joint"] = get_joint_hyps(hyps["goal-labels"], self.blocked_joint)

        hyps["method-label"] = normalise_dict(hyps["method-label"])

        none_method_prob = 1.0 - sum(hyps["method-label"].values())
        if none_method_prob > 0.0:
            hyps["method-label"]["none"] = none_method_prob

        return hyps

    def reset(self):
        self.hyps = {"goal-labels": {}, "goal-labels-joint": [], "requested-slots": {}, "method-label": {}}
        self.blocked_single = []
        self.blocked_joint = []
        self.stats = []


def clip(x):
    if x > 1:
        return 1
    if x < 0:
        return 0
    return x


def get_nonzero(x):
    m = []
    for k, p in x.items():
        if not is_zero(p):
            m.append(k)
    return m


def normalise_joint_hyps(x):
    total_p = sum([h["score"] for h in x])
    if total_p > 1.0:
        x = [{"slots": h["slots"], "score": h["score"] / total_p} for h in x]
    return x


def normalise_dict(x):
    x_items = x.items()
    total_p = sum([p for k, p in x_items])
    if total_p > 1.0:
        x_items = [(k, p / total_p) for k, p in x_items]
    return dict(x_items)


def is_zero(x):
    epsilon = 1e-9
    return math.fabs(x) < epsilon


def print_gplv3():
    print('\nThis program is free software: you can redistribute it and/or modify\n' + \
          'it under the terms of the GNU General Public License as published by\n' + \
          'the Free Software Foundation, either version 3 of the License, or\n' + \
          '(at your option) any later version.\n\n' + \
          'This program is distributed in the hope that it will be useful,\n' + \
          'but WITHOUT ANY WARRANTY; without even the implied warranty of\n' + \
          'MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n' + \
          'GNU General Public License for more details.\n\n' + \
          'You should have received a copy of the GNU General Public License\n' + \
          'along with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n')


def main():
    print_gplv3()

    parser = argparse.ArgumentParser(
        description='HWU Rule-based Dialog State Tracker Baseline V2.0\n  by Zhuoran Wang\t zhuoran.wang@hw.ac.uk\n  This version extends the work in (Wang & Lemon, SigDial 2013).', \
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze')
    parser.add_argument('--ontology', dest='ontology', action='store', metavar='JSON_FILE', required=True,
                        help='The ontology to use')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True, metavar='PATH',
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile', dest='trackfile', action='store', required=True, metavar='JSON_FILE',
                        help='File to write with tracker output')
    parser.add_argument('--original', dest='original', action='store', required=False, metavar='TRUE/FALSE',
                        help='Use the original version presented in (Wang & Lemon, SigDial 2013)')

    args = parser.parse_args()

    dataset = dataset_walker.dataset_walker(args.dataset, dataroot=args.dataroot)

    original = False
    if args.original and args.original.lower() == "true":
        original = True

    load_ontology(args.ontology)

    track_file = open(args.trackfile, "wb")
    track = {"sessions": []}
    track["dataset"] = args.dataset
    start_time = time.time()
    tracker = HWU_Tracker()

    for call in dataset:
        this_session = {"session-id": call.log["session-id"], "turns": []}
        tracker.reset()
        for turn, _ in call:
            tracker_turn = tracker.addTurn(turn, original)
            this_session["turns"].append(tracker_turn)

        track["sessions"].append(this_session)
    end_time = time.time()
    elapsed_time = end_time - start_time
    track["wall-time"] = elapsed_time

    json.dump(track, track_file, indent=4)


if __name__ == '__main__':
    main()
