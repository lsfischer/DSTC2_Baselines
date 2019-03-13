import os
import json

# A json object containing the ontology of the dataset
ontology = json.load(open("/nas/Datasets/dstc/dstc2/traindev/scripts/config/ontology_dstc2.json"))

dataset_path = "/nas/Datasets/dstc/dstc2/traindev/data/"
log_label_files_path = []

asr_indexes = []

# Get all the training "log.json" and "label.json" files
for root, subdirs, files in os.walk(dataset_path):
    if len(files) != 0:
        log_label_files_path.append((f"{root}/log.json", f"{root}/label.json"))

# Iterate over every json file in the training/dev dataset
for log_path, label_path in log_label_files_path:
    # Open the .json file
    label_file = json.load(open(label_path))
    log_file = json.load(open(log_path))

    for turn_idx in range(len(label_file["turns"])):
        label_turn = label_file["turns"][turn_idx]
        log_turn = log_file["turns"][turn_idx]

        label_utterrance = label_turn["transcription"]
        if label_utterrance == "dont care":
            label_utterrance = "don't care"

        log_user_input = log_turn["input"]

        found_matching_utterance = False

        if log_user_input:
            asr_hyps = log_user_input["live"]["asr-hyps"]

            if asr_hyps:
                for idx, asr_hyp in enumerate(asr_hyps):
                    input_utterance = asr_hyp["asr-hyp"]
                    # print(f"Label: {label_utterrance} \t Log: {input_utterance}\n\n")
                    if input_utterance == label_utterrance and not found_matching_utterance:
                        # If we found a asr hyp that matches the user utterance in the label file
                        # Then append the index in which that asr hyp occurred
                        asr_indexes.append(idx)
                        found_matching_utterance = True

        if not found_matching_utterance:
            # If we didn't find any matching asr hyp then append -1
            # to signal that there were no matching asry hyps found
            asr_indexes.append(-1)

freq = {}
for item in asr_indexes:
    if item in freq:
        freq[item] += 1
    else:
        freq[item] = 1

for key, value in freq.items():
    print("% d : % d" % (key, value))
