import os
import json

dataset_path = "/nas/Datasets/dstc/dstc2/traindev/data/"
label_files = []

for root, subdirs, files in os.walk(dataset_path):
    if len(files) != 0:
        label_files.append(f"{root}/label.json")

with open("train_data.csv", "w+") as output_file:
    for label_file_path in label_files:
        label_file = json.load(open(label_file_path))
        for turn in label_file["turns"]:

            user_uterrance = turn["transcription"].replace(',', '')
            labels = []
            for semantic_json in turn["semantics"]["json"]:
                if semantic_json["act"] == "inform":

                    slot, value = semantic_json["slots"][0]
                    if slot != "this":
                        labels.append(f"{slot}={value}")
            if labels:
                output_file.write(f"{user_uterrance},{';'.join(labels)}\n")
