from nltk.corpus import brown, stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
from operator import itemgetter 
import numpy as np

def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stop_words:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stop_words:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words=None):
    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            S[idx1][idx2] = sentence_similarity(sentences[idx1].split(), sentences[idx2].split(), stop_words)
    #print(S)
    # normalize the matrix row-wise
    for idx in range(len(S)):
        sum=S[idx].sum()
        if(sum!=0):
            S[idx] /= S[idx].sum()
    return S

def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P

def textrank(sentences, top_n, stop_words=None):
    """
    sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
    top_n = how may sentences the summary should contain
    stop_words = a list of stop_words
    """
    S = build_similarity_matrix(sentences, stop_words) 
    sentence_ranks = pagerank(S)
 
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:top_n])
    summary = itemgetter(*selected_sentences)(sentences)
    summary = list(summary)
    for index, line in enumerate(summary):
        summary[index] = line.replace("”","").replace("“","").replace(",", "").replace("&", "and").replace("'", "").replace("\"", "").replace("’","").replace("‘","")
    return summary
 
def select_sentences(text,num):
    sentences = sent_tokenize(text)
    # get the english list of stopwords
    stop_words = stopwords.words('english')
    return textrank(sentences,num, stop_words)

#print(select_sentences("I want to go to the good market. This is a bad sentence. This is a good sentence.",2))
