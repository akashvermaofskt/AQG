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

# Get a text from the Brown Corpus
text="""Mahatma Gandhi is well known as the “Father of the Nation or Bapu” because of his greatest contributions towards the independence of our country. He was the one who believed in the non-violence and unity of the people and brought spirituality in the Indian politics. He worked hard for the removal of the untouchability in the Indian society, upliftment of the backward classes in India, raised voice to develop villages for social development, inspired Indian people to use swadeshi goods and other social issues. He brought common people in front to participate in the national movement and inspired them to fight for their true freedom.

He was one of the persons who converted people’s dream of independence into truth a day through his noble ideals and supreme sacrifices. He is still remembered between us for his great works and major virtues such as non-violence, truth, love and fraternity. He was not born as great but he made himself great through his hard struggles and works. He was highly influenced by the life of the King Harischandra from the play titled as Raja Harischandra. After his schooling, he completed his law degree from England and began his career as a lawyer. He faced many difficulties in his life but continued walking as a great leader.

He started many mass movements like Non-cooperation movement in 1920, civil disobedience movement in 1930 and finally the Quit India Movement in 1942 all through the way of independence of India. After lots of struggles and works, independence of India was granted finally by the British Government. He was a very simple person who worked to remove the colour barrier and caste barrier. He also worked hard for removing the untouchability in the Indian society and named untouchables as “Harijan” means the people of God.

He was a great social reformer and Indian freedom fighter who died a day after completing his aim of life. He inspired Indian people for the manual labour and said that arrange all the resource ownself for living a simple life and becoming self-dependent. He started weaving cotton clothes through the use of Charakha in order to avoid the use of videshi goods and promote the use of Swadeshi goods among Indians.

He was a strong supporter of the agriculture and motivated people to do agriculture works. He was a spiritual man who brought spirituality to the Indian politics. He died in 1948 on 30th of January and his body was cremated at Raj Ghat, New Delhi. 30th of January is celebrated every year as the Martyr Day in India in order to pay homage to him.

"""

#print(sentences)
 
#print(len(sentences))  #  98
 

 
 
def build_similarity_matrix(sentences, stop_words=None):
    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))
 
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
 
            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
 
    # normalize the matrix row-wise
    for idx in range(len(S)):
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
    return list(summary)
 
def select_sentences(text,num):
    sentences = sent_tokenize(text)
    # get the english list of stopwords
    stop_words = stopwords.words('english')
    return textrank(sentences,num, stop_words)
