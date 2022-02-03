
import argparse
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from models import EmModel
import numpy as np

from utils import TOPICS

parser = argparse.ArgumentParser(description='Get args from bash')
parser.add_argument('development_set_filename', metavar='D', type=str, nargs=1,
                    help='development set filename, ends with .txt')

# read from txt and split the documents into single words

if __name__ == '__main__':
    args = parser.parse_args()
    em_model = EmModel(dataset_file_path = args.development_set_filename[0])
    soft_clustering = em_model.EM()
    confusion_matrix = em_model.create_confusion_matrix()
    confusion_matrix.to_html(buf= open('confusion_matrix.html','w'))
    accuracy = em_model.calculate_accuracy()
    print(accuracy)
    for i in range(9):
        cluster_array = confusion_matrix.iloc[i,0:9].to_numpy()
        plt.bar(x = TOPICS,height = cluster_array)
        plt.title(f'Cluster {i}, topic:{TOPICS[np.argmax(cluster_array)]}')
        plt.show()
    plt.plot(range(len(em_model.log_likelihood_record)), em_model.log_likelihood_record,scaley='log')
    plt.title('Log Likelihood vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('log likelihood')
    plt.show()
    plt.plot(range(len(em_model.perplexity_record)), em_model.perplexity_record)
    plt.title('Perplexity vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('mean perplexity per word')
    plt.show()
