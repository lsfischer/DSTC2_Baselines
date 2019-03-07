import copy
import pickle
import string
import numpy as np
import tensorflow as tf
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from bert_serving.client import BertClient
from trackers.abstract_tracker import AbstractTracker
from deep_contextual_bandits.bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from deep_contextual_bandits.bandits.core.contextual_bandit import ContextualBandit


class BanditTrackerTF(AbstractTracker):

    def __init__(self, ontology):
        super(BanditTrackerTF, self).__init__(ontology)
        self.bc = BertClient(check_version=False, check_length=False)

        self.food_dataset, self.food_opt_rewards, self.food_opt_actions, _, _ = self.get_dataset(
            pickle.load(open("../training_data/train_data_food_v2", "rb")))

        self.area_dataset, self.area_opt_rewards, self.area_opt_actions, _, _ = self.get_dataset(
            pickle.load(open("../training_data/train_data_area_v2", "rb")))

        self.price_dataset, self.price_opt_rewards, self.price_opt_actions, _, _ = self.get_dataset(
            pickle.load(open("../training_data/train_data_pricerange_v2", "rb")))

    def get_dataset(self, data_object):
        # convert to np_array
        data_object["features"] = np.array(data_object["features"])
        data_object["labels"] = np.array(data_object["labels"])

        num_actions = 2  # Actions are : Update state, Do Not Update state
        context_dim = 2049
        noise_stds = [0.01 * (i + 1) for i in range(num_actions)]

        betas = np.random.uniform(-1, 1, (context_dim, num_actions))
        betas /= np.linalg.norm(betas, axis=0)

        rewards = np.dot(data_object["features"], betas)
        opt_actions = np.argmax(rewards, axis=1)
        rewards += np.random.normal(scale=noise_stds, size=rewards.shape)
        opt_rewards = np.array([rewards[i, act] for i, act in enumerate(opt_actions)])
        return np.hstack((data_object["features"], rewards)), opt_rewards, opt_actions, num_actions, context_dim

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

        # English stopwords set with punctuation
        stop = stopwords.words('english') + list(string.punctuation)

        # Tokenize the best hypothesis on the whitespaces
        tkns = word_tokenize(best_asr_hyp)

        # Remove stop words and also shingle the tokens
        processed_hyp = [word for word in tkns if
                         word not in stop]  # + [tup[0] + " " + tup[1] for tup in ngrams(tkns, 2)]

        # Manually change from "moderately"/"affordable" to "moderate" and "cheaper" to "cheap"
        for idx, word in enumerate(processed_hyp):
            if word == "moderately" or word == "affordable":
                processed_hyp[idx] = "moderate"
            if word == "cheaper":
                processed_hyp[idx] = "cheap"

        num_actions = 2  # Possible actions : Update state, Do Not update sate

        context_dim = 2049  # Concatenation of all features

        hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                                     context_dim=context_dim,
                                                     a0=6,
                                                     b0=6,
                                                     lambda_prior=0.25,
                                                     initial_pulls=2)

        algo = LinearFullPosteriorSampling('LinFullPost', hparams_linear)

        food_bandit = ContextualBandit(context_dim, num_actions)

        food_bandit.feed_data()


        self.hyps = hyps
        return self.hyps
