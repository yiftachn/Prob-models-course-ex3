# Yiftach Neuman Itai Mondshine 208305359 207814724

import argparse
import logging
import pickle
from matplotlib import pyplot
import pandas as pd
from modles import EmModel
import numpy as np

parser = argparse.ArgumentParser(description='Get args from bash')
parser.add_argument('development_set_filename', metavar='D', type=str, nargs=1,
                    help='development set filename, ends with .txt')

# read from txt and split the documents into single words

if __name__ == '__main__':
    args = parser.parse_args()

    # on the first run use these two lines

    em_model = EmModel(dataset_file_path = args.development_set_filename[0])
    # instead of this line
    soft_clustering = em_model.EM()
    em_model.hard_cluster()
    em_model.create_confusion_matrix()
    # pyplot.plot(range(len(em_model.perplexity_record)), em_model.perplexity_record)
    pyplot.plot(range(len(em_model.perplexity_record)), em_model.log_likelihood_record)
    pyplot.show()
    pyplot.plot(range(len(em_model.perplexity_record)), np.exp(em_model.perplexity_record))
    pyplot.show()
    pyplot.plot(range(len(em_model.words_perplexity_record)), em_model.words_perplexity_record)
    pyplot.show()

