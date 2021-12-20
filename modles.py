from utils import split_input
import math

class EM:
    def __init__(self, documents_dict):
        self._documents = documents_dict
        self.distribution_in_documents = []
        self.alpha = []
        self.scores = []




    def save(self):
        pass


    def generateMatrix(self):
        pass



    def perplexity(self):
        pass


    def eStep(self):
        for doc in self._documents:
            z = math.log(self.alpha[i]  + sum(doc[))
            z = math.log(self.alpha[i]) + sum(doc[i][word] * math.log(P[i][word])
                                              for i in range(9))]
            m = max(z)
            denominator = sum(map(lambda j : math.exp(z[j] - m), indexs))
            return_val = z / denominator

        z = log



    def mStep(self):
        print("Starting m step")

        pass








    def em_initialize(self):
        pass





    def eStep(self):
        pass



