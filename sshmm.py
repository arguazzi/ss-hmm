#!/usr/bin/env python2
# Copyright Alessandro Guazzi 2018
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
import numpy as np

class SemiSupervisedHMM:

    def __init__(self, seq_labelled, seq_unlabelled, seq_labelled_len, seq_unlabelled_len, seq_labels, n_states, semi_supervised=True, verbose=False):
        """
        Method extending the hmmlearn package for Hidden Markov Models with
        the semi-supervised approach published in "Semi-Supervised Sequence
        Classification with HMMs" (Zhong, 2005. DOI:10.1.1.321.7760).
        In the case for which an HMM needs to be trained without using unlabelled
        data, it can be called with the parameter semi_supervised set to False.

        :param seq_labelled: 1D numpy array containing all labelled sequences
        :param seq_unlabelled: 1D numpy array containing all unlabelled sequences
        :param seq_labelled_len: 1D numpy array containing the lengths of each labelled sequence
        :param seq_unlabelled_len: 1D numpy array containing the lengths of each unlabelled sequence
        :param seq_labels: 1D numpy array containing the label for each sequence (K labels 0 to K-1)
        :param n_states: 1D numpy array with the number of states for the HMMs corresponding to each class
        :param semi_supervised: boolean, set to False if only want to use labelled arrays (default True)
        :param verbose: boolean, set to True for additional information (default False)
        """


        self.seq_l = seq_labelled
        self.seq_u = seq_unlabelled

        self.len_l = seq_labelled_len.astype(int)
        self.len_u = seq_unlabelled_len.astype(int)

        self.lbl_l = seq_labels.astype(int)
        self.lbl_u = -1*np.ones(len(self.len_u))

        self.n_states = n_states.astype(int)
        self.is_ss = semi_supervised
        self.is_verbose = verbose

        self.classes = np.unique(self.lbl_l)
        self.M = []
        self.K = []

        self.seq_a = np.append(self.seq_l, self.seq_u)
        self.len_a = np.append(self.len_l, self.len_u)

        # Initialise KMeans and GaussianHMMs for each class
        for c in self.classes:
            self.M.append(GaussianHMM(n_components=n_states[c]))
            self.K.append(KMeans(n_clusters=n_states[c], random_state=0))

    def run(self):
        """
        Run the train + assign steps. The code will iteratively train the HMMs for the class for all available data,
        starting with the labelled data. All unlabelled data will then be assigned to the class with the highest log
        likelihood. The HMMs will be retrained, initialised with the means of all the data, and the process is repeated
        until the classification becomes consistent with itself or after N iterations.
        After the any training step, the assign step essentially acts as a classifier, so it can be used on any new data
        for classification purposes.
        """
        n_iterations = 10

        self.train(self.seq_l, self.len_l, self.lbl_l)
        self.assign(self.seq_u, self.len_u)

        i = 0
        is_not_converged = True
        while (i < n_iterations) & is_not_converged & self.is_ss:
            i += 1
            self.lbl_o = self.lbl_u
            self.lbl_a = np.append(self.lbl_l, self.lbl_u)

            self.train(self.seq_a, self.len_a, self.lbl_a)
            self.assign(self.seq_u, self.len_u)

            d = np.sum(self.lbl_u != self.lbl_o)

            if self.is_verbose:
                print "Iteration "+str(i)+", error: "+str(d)

            if d==0:
                is_not_converged = False

    def get_ix(self, lens, labels, c):
        i = []
        for l in range(0, len(labels)):
            i = np.append(i, np.ones(lens[l])*(labels[l]==c))
        return i.astype(bool)

    def train(self, seq, lens, labels):
        """
        Training step.
        :param seq: 1-D numpy array containing the sequences to be trained on
        :param lens: 1-D numpy array containing the lengths of each sequence
        :param labels: 1-D numpy array containing the labels of each sequence
        """
        for c in self.classes:
            # get sequence of interest
            train = seq[self.get_ix(lens, labels, c)].reshape(-1,1)
            # obtain estimate of class means
            self.K[c].fit(train)
            # initialise the Gaussian HMM with class means
            self.M[c] = GaussianHMM(n_components=self.n_states[c], init_params='m')
            self.M[c].means_ = self.K[c].cluster_centers_
            # fit HMM to data
            self.M[c].fit(train, lens[labels==c])

    def assign(self, seq, lens):
        """
        Classification step (outputs labels for each sequence to lbl_u)
        :param seq: 1-D numpy array containing the unlabelled sequences to be classified
        :param lens: 1-D numpy array containing the lengths of each sequence
        """
        self.lbl_u = -1*np.ones(len(lens))
        for i in  range(0, len(lens)):
            seq_i = seq[np.sum(lens[:i+1])-lens[i]:np.sum(lens[:i+1])].reshape(-1, 1)
            ref = -np.inf
            for c in self.classes:
                s = self.M[c].score(seq_i)
                if s > ref:
                    ref = s
                    u_class = c
            self.lbl_u[i] = u_class
