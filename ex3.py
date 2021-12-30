

# Yiftach Neuman Itai Mondshine 208305359 207814724

import argparse
import logging
import pickle

import pandas as pd
from modles import EmModel

parser = argparse.ArgumentParser(description='Get args from bash')
parser.add_argument('development_set_filename', metavar='D', type=str, nargs=1,
                    help='development set filename, ends with .txt')


    # read from txt and split the documents into single words

if __name__ == '__main__':
    args = parser.parse_args()

    # on the first run use these two lines

    # em_model = EmModel(dataset_file_path = args.development_set_filename[0])
    # pickle.dump(em_model,open('em_model.pkl','wb'))

    #instead of this line
    em_model = pickle.load(open('em_model.pkl','rb'))
    soft_clustering = em_model.EM()
