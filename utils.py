from collections import Counter
from typing import List, Set, Tuple

import numba
import numpy as np
import pandas as pd

TOPICS = open('./dataset/topics.txt').read().split()
GROUPS_NUMBER = 9


class Dataset:
    def __init__(self, path_to_dataset_file: str):
        self.dataset_file_path = path_to_dataset_file
        self.words: List[str] = sorted(self._get_words_list())
        self.documents_count = self._get_documents_count()
        self.vocabulary_length = len(self.words)
        self.words_counter = Counter(self.words)
        self.X, self.y = self._create_text_words_df(path_to_dataset_file)
        self.X_transpose = self.X.T # vocabulary * documents

    def _get_words_list(self) -> List[str]:
        words = []
        # Build an array with all the words in the file
        with open(self.dataset_file_path, 'r') as file_str:
            for line in file_str:
                if line[:6] != '<TRAIN':
                    for word in line.split():
                        words.append(word)
        file_str.close()
        words_set = set(words)
        words_counter = Counter(words)
        for word in words_counter.keys():
            if words_counter[word] <= 3:
                words_set.remove(word)
        word_set_as_tuple = list(words_set)
        return word_set_as_tuple

    def _get_documents_count(self):
        documents_count = 0
        # Build an array with all the words in the file
        with open(self.dataset_file_path, 'r') as file_str:
            for line in file_str:
                if line[:6] != '<TRAIN' and line != '\n':
                    documents_count = documents_count + 1
        return documents_count
    @numba.jit
    def _create_text_words_df(self, dataset_file_path: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
        texts = np.zeros((self.documents_count, self.vocabulary_length))
        topics = np.zeros((self.documents_count, len(TOPICS)))
        index = 0
        file_str = open(dataset_file_path, 'r')
        for line in file_str:
            if line[:6] == '<TRAIN':
                topics_names = line.split()[2:]
                for topic in topics_names:
                    if topic[-1] == '>':
                        topic = topic[:-1]
                    topics[index][get_topic_index(topic)] = 1
            elif line != '\n':
                words_count_in_documents = (Counter(line.split()))
                for word in words_count_in_documents.keys():
                    if self.words_counter[word] > 0:
                        texts[index][self.words.index(word)] = words_count_in_documents[word]
                index = index + 1
        text_df = pd.DataFrame(texts, columns=list(range(0, self.vocabulary_length)))
        topics_df = pd.DataFrame(topics, columns=['y' + str(x) for x in range(0, len(TOPICS))])
        return (text_df, topics_df)


def get_topic_index(topic_name: str) -> int:
    return TOPICS.index(topic_name)
