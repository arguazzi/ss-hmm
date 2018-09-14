#!/usr/bin/env python2
# Copyright Alessandro Guazzi 2018
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
import numpy as np

class SemiSupervisedHMM():
    """ SemiSupervisedHMM
    Method extending the hmmlearn package for Hidden Markov Models with
    the semi-supervised approach published in "Semi-Supervised Sequence
    Classification with HMMs" (Zhong, 2005. DOI:10.1.1.321.7760).

    In the case for which an HMM needs to be trained without using unlabelled
    data, it can be called with the parameter semi_supervised set to False.
    """

    def __init__(self, seq_labelled, seq_unlabelled, seq_labelled_len,
    seq_unlabelled_len, seq_labels, n_states, semi_supervised=True, verbose=False):

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
        self.kmeans = KMeans(n_clusters=2, random_state=0)

        self.seq_a = np.append(self.seq_l, self.seq_u)
        self.len_a = np.append(self.len_l, self.len_u)

        for c in self.classes:
            self.M.append(GaussianHMM(n_components=n_states[c]))

    def run(self):
        self.train(self.seq_l, self.len_l, self.lbl_l)
        self.assign(self.seq_u, self.len_u)

        i = 0
        is_not_converged = True
        while (i < 10) & is_not_converged & self.is_ss:
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
        for c in self.classes:
            # get sequence of interest
            train = seq[self.get_ix(lens, labels, c)].reshape(-1,1)
            # obtain estimate of class means
            self.kmeans.fit(train)
            # initialise the Gaussian HMM with class means
            self.M[c] = GaussianHMM(n_components=self.n_states[c], init_params='m')
            self.M[c].means_ = self.kmeans.cluster_centers_
            # fit HMM to data
            self.M[c].fit(train, lens[labels==c])

    def assign(self, seq, lens):
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
