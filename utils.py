from collections import Counter
GROUPS = 9


def split_input(document: list) -> list:
    list_of_groups = [[] for i in range(GROUPS)]
    for i in range(len(document)):
        list_of_groups[i % 9].append(document[i])
    return list_of_groups


def add_distribution_save_to_dict(documents: list) -> list:
    documnets_dict = {}
    for i in range(len(documents)):
        if len(documents[i]) != 0:
            doc_count = Counter(documents[i][1])
            documnets_dict[i] = {'document': documents[i], 'doc_counter': doc_count}
    return documnets_dict


def read_file(file):
    document = []
    documents = []
    with open(file) as f:
        for line in f:
            if line[:6] != '<TRAIN' and line[:5] != '<TEST':
                document.append(line.split())
            else:
                documents.append(document)
                document = []
    f.close()
    return documents


