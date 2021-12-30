import pickle
from typing import List
from collections import Counter

import numba.core.codegen

from utils import Dataset, get_topic_index
import numpy as np
import pandas as pd
from scipy import sparse
from utils import GROUPS_NUMBER
import math

K = 10
EPSILON = 0.00001


class EmModel:
    def __init__(self, dataset_file_path: str):
        #todo: calculate documents perplexity

        # save to pickle in debug mode then switch to pickle load - will be much faset.dont froget to uncomment this!!
# todo: make sure you dont forget to delete the pickling when handing in the assigmnet
        self.dataset = Dataset(dataset_file_path)
        # self.dataset = pickle.load(open('dataset.pkl', 'rb'))
        self.cluster_probs = np.zeros((self.dataset.documents_count, GROUPS_NUMBER))
        self.alpha = np.zeros((1, GROUPS_NUMBER))
        self.words_probs = np.zeros((self.dataset.vocabulary_length, GROUPS_NUMBER))
        self.log_likelihood_record = []
        self.perplexity_record = []
        self.words_perplexity_record = []
        self.lmbda = 0.9
        self.threshold = 0.000001
        self._initialize()

    def EM(self, iterations: int = 1000):
        new_likelihood = -1 * 10000 * 1000000
        old_likelihood = -1 * np.inf

        while (old_likelihood + self.threshold < new_likelihood):
            assert new_likelihood > old_likelihood

            old_likelihood = new_likelihood
            self.e_step()
            self.m_step()
            new_likelihood = self.log_likelihood()
            self.words_perplexity_record.append(self._calculate_words_perplexity())
            self.log_likelihood_record.append(new_likelihood)
            self.perplexity_record.append(new_likelihood * (1 / self.dataset.documents_count))
            # print(f'new L: {format(new_likelihood, ".6e")}, old L: {format(new_likelihood, ".6e")}')

    @numba.jit
    def _initialize(self):
        for i in range(0, self.dataset.documents_count):
            self.cluster_probs[i, i % GROUPS_NUMBER] = 1
        self.alpha = self.cluster_probs.sum(axis=0) / self.dataset.documents_count
        self.m_step()

    @numba.jit
    def e_step(self):
        # checked
        Z = np.log(self.alpha) + self.dataset.X @ np.log(self.words_probs)
        Z_stable = _stabilize_numerically(Z)
        denominator = Z_stable.sum(axis=1)
        self.cluster_probs = (Z_stable.T / denominator).T
        self.alpha = self.cluster_probs.sum(axis=0) / self.cluster_probs.shape[0]

    @numba.jit
    def m_step(self):
        self.alpha = np.where(self.alpha > 0, self.alpha, EPSILON)
        self.alpha = self.alpha / self.alpha.sum()
        nominator = self.dataset.X_transpose @ self.cluster_probs + self.lmbda
        denominator = self.dataset.X_transpose.sum(
            axis=0) @ self.cluster_probs + self.dataset.vocabulary_length * self.lmbda
        self.words_probs = nominator / denominator

    def log_likelihood(self) -> np.float64:
        Z = np.log(self.alpha) + self.dataset.X @ np.log(self.words_probs)
        self._m = np.max(Z, axis=1)
        Z_stable = _stabilize_numerically(Z)
        documents_probs = Z_stable
        return (np.log(documents_probs.sum(axis=1)) + self._m).sum()

    def _calculate_words_perplexity(self) -> np.ndarray:
        mean_words_perplexity = np.multiply(self.words_probs, self.alpha).sum(axis=1)
        words_perplexity = np.log(mean_words_perplexity).sum()
        normalized_words_perplexity = -1* words_perplexity/self.dataset.vocabulary_length
        return np.exp(normalized_words_perplexity)

    def hard_cluster(self):
        # just use np.argmax ....
        pass


@numba.jit
def _stabilize_numerically(Z: np.ndarray) -> np.ndarray:
    m = np.max(Z, axis=1)
    # zero if Zi is unstable, 1 if its stable.
    small_z = ((Z.T - m) >= -K).astype(int).T
    new_Z = np.multiply(np.exp(Z.T - m).T, small_z)
    return new_Z
