# Yiftach Neuman Itai Mondshine 208305359 207814724

import argparse
import logging
from utils import read_file, split_input, save_distributions

parser = argparse.ArgumentParser(description='Get args from bash')
parser.add_argument('development_set_filename', metavar='D', type=str, nargs=1,
                    help='development set filename, ends with .txt')


if __name__ == '__main__':
    # read from txt and split the documents into single words
    args = parser.parse_args()
    _documents = read_file(args.development_set_filename[0])
    documnets_dict = save_distributions(_documents)
    print(documnets_dict[2]['doc_counter'])



