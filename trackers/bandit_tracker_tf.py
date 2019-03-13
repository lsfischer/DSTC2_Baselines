import copy
import pickle
import string
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from bert_serving.client import BertClient
from sklearn.preprocessing import normalize
from trackers.abstract_tracker import AbstractTracker
from bandits.core.contextual_bandit import ContextualBandit
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling


class BanditTrackerTF(AbstractTracker):

    def __init__(self, ontology):
        super(BanditTrackerTF, self).__init__(ontology)
        self.bc = BertClient(check_version=False, check_length=False, ip="compute-0-1.local")

        self.num_actions = 2  # Possible actions : Update state, Do Not update sate

        self.context_dim = 2049  # Concatenation of all features

        # Define hyper parameters to use in Contextual Bandit Algorithm
        hparams_linear = tf.contrib.training.HParams(num_actions=self.num_actions,
                                                     context_dim=self.context_dim,
                                                     a0=6,
                                                     b0=6,
                                                     lambda_prior=0.25,
                                                     initial_pulls=2)

        hparams_dropout = tf.contrib.training.HParams(num_actions=self.num_actions,
                                                      context_dim=self.context_dim,
                                                      init_scale=0.3,
                                                      activation=tf.nn.relu,
                                                      layer_sizes=[50],
                                                      batch_size=512,
                                                      activate_decay=True,
                                                      initial_lr=0.1,
                                                      max_grad_norm=5.0,
                                                      show_training=False,
                                                      freq_summary=1000,
                                                      buffer_s=-1,
                                                      initial_pulls=2,
                                                      optimizer='RMS',
                                                      reset_lr=True,
                                                      lr_decay_rate=0.5,
                                                      training_freq=50,
                                                      training_epochs=100,
                                                      use_dropout=True,
                                                      keep_prob=0.80)

        self.food_dataset, self.food_opt_rewards, self.food_opt_actions, _, _ = self.get_dataset(
            pickle.load(open("/home/l.fischer/DSTC2_Baselines/training_data/train_data_food_v2", "rb")))

        self.area_dataset, self.area_opt_rewards, self.area_opt_actions, _, _ = self.get_dataset(
            pickle.load(open("/home/l.fischer/DSTC2_Baselines/training_data/train_data_area_v2", "rb")))

        self.price_dataset, self.price_opt_rewards, self.price_opt_actions, _, _ = self.get_dataset(
            pickle.load(open("/home/l.fischer/DSTC2_Baselines/training_data/train_data_pricerange_v2", "rb")))

        self.food_algo = PosteriorBNNSampling('Dropout_food', hparams_dropout, 'RMSProp')

        self.area_algo = PosteriorBNNSampling('Dropout_area', hparams_dropout, 'RMSProp')

        self.price_algo = PosteriorBNNSampling('Dropout_price', hparams_dropout, 'RMSProp')

        self.train()

    def is_word_in_ontology(self, word, slot_type="food"):
        """
        Returns a boolean saying whether a given word is present in the ontology
        :param word: The word to check if it's in the ontology
        :param slot_type: The type of slot in which to check if the word is present
        :return: a Boolean saying whether a given word is present in the ontology
        """

        if slot_type == "food":
            return int(word in self.ontology["informable"]["food"])

        elif slot_type == "area":
            return int(word in self.ontology["informable"]["area"])

        else:
            return int(word in self.ontology["informable"]["pricerange"])

    def get_dataset(self, data_object):
        # convert to np_array
        data_object["features"] = normalize(np.array(data_object["features"]), norm="l1")
        data_object["labels"] = np.array(data_object["labels"])

        rewards = np.array([(0, 1) if label else (1, 0) for label in data_object["labels"]])

        num_actions = 2  # Actions are : Update state, Do Not Update state
        context_dim = 2049
        # noise_stds = [0.01 * (i + 1) for i in range(num_actions)]

        betas = np.random.uniform(-1, 1, (context_dim, num_actions))
        betas /= np.linalg.norm(betas, axis=0)

        # rewards = np.random.randint(2, size=(10000, 2))
        opt_actions = np.argmax(rewards, axis=1)

        opt_rewards = np.array([rewards[i, act] for i, act in enumerate(opt_actions)])
        return np.hstack((data_object["features"], rewards)), opt_rewards, opt_actions, num_actions, context_dim

    def train(self):

        # Instantiate Contextual Bandit Object
        food_bandit = ContextualBandit(self.context_dim, self.num_actions)

        food_bandit.feed_data(self.food_dataset)

        # Training food bandit classifier

        print("Training food")
        for i in tqdm(range(self.food_dataset.shape[0])):
            context = food_bandit.context(i)
            action = self.food_algo.action(context)
            reward = food_bandit.reward(i, action)

            self.food_algo.update(context, action, reward)

        # Instantiate Contextual Bandit Object
        area_bandit = ContextualBandit(self.context_dim, self.num_actions)

        area_bandit.feed_data(self.area_dataset)

        # Training area bandit classifier

        print("Training area")
        for i in tqdm(range(self.area_dataset.shape[0])):
            context = area_bandit.context(i)
            action = self.area_algo.action(context)
            reward = area_bandit.reward(i, action)

            self.area_algo.update(context, action, reward)

        # Instantiate Contextual Bandit Object
        price_bandit = ContextualBandit(self.context_dim, self.num_actions)

        price_bandit.feed_data(self.price_dataset)

        # Training price bandit classifier

        print("Training price")
        for i in tqdm(range(self.price_dataset.shape[0])):
            context = price_bandit.context(i)
            action = self.price_algo.action(context)
            reward = price_bandit.reward(i, action)

            self.price_algo.update(context, action, reward)

        print("Training Complete")

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

        if processed_hyp:

            # Create an embedding of the user utterance using BERT
            sentence_embedding = np.array(self.bc.encode([best_asr_hyp]))[0]

            # Iterate through all the words in the user utterance to obtain the features needed
            for word in processed_hyp:

                # Create and embedding of the word, being iterated, using BERT
                word_embedding = np.array(self.bc.encode([word]))[0]

                # Check whether the current word is present in the ontology, in one of the slot types
                word_in_food_ontology = [self.is_word_in_ontology(word, slot_type="food")]
                word_in_area_ontology = [self.is_word_in_ontology(word, slot_type="area")]
                word_in_price_ontology = [self.is_word_in_ontology(word, slot_type="price")]

                # Concatenate the features together (the result is a vector of size 2049)
                food_features = np.concatenate((word_embedding, sentence_embedding, word_in_food_ontology))
                area_features = np.concatenate((word_embedding, sentence_embedding, word_in_area_ontology))
                price_features = np.concatenate((word_embedding, sentence_embedding, word_in_price_ontology))

                # Decide whether the current word should update one (or more) of the slot types
                update_food_slot = self.food_algo.action(food_features)
                update_area_slot = self.area_algo.action(area_features)
                update_price_slot = self.price_algo.action(price_features)

                if update_food_slot:
                    goal_stats["food"][word] += 1.0

                if update_area_slot:
                    goal_stats["area"][word] += 1.0

                if update_price_slot:
                    goal_stats["pricerange"][word] += 1.0

            # pick top values for each slot
        super(BanditTrackerTF, self).fill_goal_labels(goal_stats, hyps)
        super(BanditTrackerTF, self).fill_joint_goals(hyps)

        self.hyps = hyps
        return self.hyps
