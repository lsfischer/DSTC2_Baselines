import copy
import numpy as np
import string
from abstract_tracker import AbstractTracker
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams


class BertTracker(AbstractTracker):

    def __init__(self, ontology):
        """
        Initializes an instance of this class
        :param ontology: JSON object containing the ontology of the task
        """
        super(BertTracker, self).__init__(ontology)
        self.bc = BertClient(check_version=False)  # Create a BERT client instance
        self.knowledge_base, self.encoded_kb = self.encode_ontology()

    def addTurn(self, turn):
        """
        Adds a turn to this tracker
        :param turn: The turn to process and add
        :return: A hypothesis of the current state of the dialog
        """

        hyps = copy.deepcopy(self.hyps)

        # Obtaining the best hypothesis from the ASR module
        best_asr_hyp = turn['input']["live"]['asr-hyps'][0]["asr-hyp"]

        # English stopwords set with punctuation
        stop = stopwords.words('english') + list(string.punctuation)

        # Tokenize the best hypothesis on the whitespaces
        tkns = word_tokenize(best_asr_hyp)

        # Remove stop words and also shingle the tokens
        processed_hyp = [word for word in tkns if word not in stop] + [tup[0] + " " + tup[1] for tup in ngrams(tkns, 2)]

        # Manually change from "moderately"/"affordable" to "moderate" and "cheaper" to "cheap"
        for idx, word in enumerate(processed_hyp):
            if word == "moderately" or word == "affordable":
                processed_hyp[idx] = "moderate"
            if word == "cheaper":
                processed_hyp[idx] = "cheap"

        if processed_hyp:

            # Obtain the ontology information
            pricerange_options = self.ontology["informable"]["pricerange"]
            food_options = self.ontology["informable"]["food"]
            area_options = self.ontology["informable"]["area"]

            state_updated = False

            # SIMPLE Matching
            # Iterate through all the words in the best asr hypothesis
            # If the word is present in the ontology update that slot with the word
            for hyp_word in processed_hyp:

                if hyp_word in food_options:
                    hyps["goal-labels"]["food"] = {
                        hyp_word: 1.0
                    }
                    state_updated = True

                if hyp_word in area_options:
                    hyps["goal-labels"]["area"] = {
                        hyp_word: 1.0
                    }
                    state_updated = True

                if hyp_word in pricerange_options:
                    hyps["goal-labels"]["pricerange"] = {
                        hyp_word: 1.0
                    }
                    state_updated = True

            # If this simple matching was not able to match anything then we will use BERT w/ cosine-similarity
            if not state_updated:

                # Use BERT to encode all the words in the sentence
                encoded_hyp = np.array(self.bc.encode(processed_hyp))

                # Use the cosine sim between the previous encoding and the encoded knowledge base
                cosine_sim = cosine_similarity(encoded_hyp, self.encoded_kb)

                for idx, sub_arr in enumerate(cosine_sim):

                    # For every word in the sentence obtain the word in the KB that maximizes the cosine sim
                    argmax_index = np.argmax(sub_arr)

                    # assuming that if it's lower than 0.97 then it's probably a mistake
                    # (Not many cases have 0.97 cosine sim, maybe none actually)
                    if sub_arr[argmax_index] >= 0.97:

                        kb_word = self.knowledge_base[argmax_index]
                        print(f"BERT: Word in query: {processed_hyp[idx]} \t matched with {kb_word}")

                        if kb_word in food_options:
                            hyps["goal-labels"]["food"] = {
                                kb_word: 1.0
                            }

                        if kb_word in area_options:
                            hyps["goal-labels"]["area"] = {
                                kb_word: 1.0
                            }

                        if kb_word in pricerange_options:
                            hyps["goal-labels"]["pricerange"] = {
                                kb_word: 1.0
                            }

            super(BertTracker, self).fill_joint_goals(hyps)

        self.hyps = hyps
        return self.hyps

    def encode_ontology(self):
        """Encodes the ontology JSON using BERT"""

        ontology = copy.deepcopy(self.ontology)

        pricerange_options = ontology["informable"]["pricerange"]
        food_options = ontology["informable"]["food"]
        area_options = ontology["informable"]["area"]

        knowledge_base = food_options + area_options + pricerange_options

        return knowledge_base, np.array(self.bc.encode(knowledge_base))
