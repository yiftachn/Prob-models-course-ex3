

# Yiftach Neuman Itai Mondshine 208305359 207814724

import argparse
import logging
import pandas as pd
from modles import EmModel

parser = argparse.ArgumentParser(description='Get args from bash')
parser.add_argument('development_set_filename', metavar='D', type=str, nargs=1,
                    help='development set filename, ends with .txt')


    # read from txt and split the documents into single words

if __name__ == '__main__':
    args = parser.parse_args()
    em_model = EmModel(dataset_file_path = args.development_set_filename[0])

    soft_clustering = em_model.cluster_data()
    prediction = em_model.classify_data()
