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
        self.dataset = Dataset(dataset_file_path)
        self.cluster_probs = np.zeros((self.dataset.documents_count, GROUPS_NUMBER))
        self.alpha = np.zeros((1, GROUPS_NUMBER))
        self.documents_probs = np.ones((self.dataset.documents_count, GROUPS_NUMBER))
        self.words_probs = np.zeros((self.dataset.vocabulary_length, GROUPS_NUMBER))
        self.data_likelihood = -1 * 10000 * 10000
        self.lmbda = 0.00001
        self._initialize()

    # todo: convert to sparse
    def EM(self):
        new_likelihood = -1 * 10000 * 10000
        # while (self.data_likelihood + 0.0001 < new_likelihood):
        for i in range(100):
            self.data_likelihood = new_likelihood
            self.e_step()
            self.m_step()
            self.calculate_documents_probs()
            new_likelihood = self.log_likelihood()
            print(f'new L: {new_likelihood}, old L: {self.data_likelihood}')

    def _initialize(self):
        for i in range(0, self.dataset.documents_count):
            self.cluster_probs[i][i % 9] = 1
        self.alpha = self.cluster_probs.sum(axis=0) / self.dataset.documents_count
        self.m_step()

    def e_step(self):
        Z = np.log(self.alpha) + self.dataset.X @ np.log(self.words_probs)
        Z_stable = _stabilize_numerically(Z)
        denominator = Z_stable.sum().sum()
        self.cluster_probs = Z_stable / denominator
        self.alpha = self.cluster_probs.sum(axis=0)

    def m_step(self):
        self.alpha = np.where(self.alpha > 0, self.alpha, EPSILON)
        self.alpha = self.alpha / self.alpha.sum()
        nominator = self.dataset.X_transpose @ self.cluster_probs + self.lmbda
        denominator = self.dataset.X_transpose.sum(
            axis=0) @ self.cluster_probs + self.dataset.vocabulary_length * self.lmbda
        self.words_probs = nominator / denominator

    def calculate_documents_probs(self):
        Z = np.log(self.alpha) + self.dataset.X @ np.log(self.words_probs)
        self._m = np.max(Z, axis=0)
        Z_stable = _stabilize_numerically(Z)
        self.documents_probs = Z_stable

    def log_likelihood(self) -> float:
        return (np.log(self.documents_probs.sum()) + self._m).sum()


def _stabilize_numerically(Z: np.ndarray) -> np.ndarray:
    m = np.max(Z, axis=0)
    # zero if Zi is unstable, 1 if its stable.
    small_z = ((Z - m + K) >= 0).astype(int)
    new_Z = np.multiply(np.exp(Z - m), small_z)
    return new_Z
