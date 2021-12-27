from collections import Counter
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

GROUPS_NUMBER = 9
topics = open('./dataset/topics.txt').read().split()


class Dataset:
    def __init__(self, path_to_dataset_file: str):
        self.dataset_file_path = path_to_dataset_file
        self.words: List[str] = self._get_words_list()
        self.documents_count = self._get_documents_count()
        self.vocabulary_length = len(self.words)
        self.words_counter = Counter(self.words)
        self.dataset = self._create_text_words_df(path_to_dataset_file)

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
        words_counter = Counter(word)
        for word in words_counter.keys():
            if words_counter[word] < 3:
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

    def _create_text_words_df(self,dataset_file_path:str) -> pd.DataFrame:
        texts = np.zeros((self.documents_count,self.vocabulary_length))
        topics = np.zeros((self.documents_count,GROUPS_NUMBER))
        index = 0
        with open(dataset_file_path,'r') as file_str:
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
        texts_and_topics = np.concatenate((texts,topics),axis = 1)

        df = pd.DataFrame(texts_and_topics,columns=list(range(0,self.vocabulary_length))+['y'+str(x) for x in range(0,GROUPS_NUMBER)])
        return df

def get_topic_index(topic_name: str) -> int:
    return topics.index(topic_name)


