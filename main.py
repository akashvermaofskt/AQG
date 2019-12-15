from SentenceSelector import select_sentences
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import nltk
from textblob import TextBlob

text="""An old man lived with his four sons in a village. The old man was worried. His sons were always quarrelling with each other. He had tried telling them many times to avoid fighting. But his sons would not listen to him.
One day, he called his four sons. He gave them a small bundle of sticks, and asked them to break the bundle into two. The bundle was made of four sticks. “It’s child’s play,” said the eldest son. He took the bundle and tried to break it. He was surprised that the sticks in the bundle remained intact. He used more force. He tried again and again. He started panting for breath. The bundle would not break. He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle. He asked each brother to take one stick and try to break it. Each of the sons took a stick in hand. In no time, the sticks were bent and broken.
“A single stick is easily broken. If four sticks come together it is impossible to break them,” said the old man, giving his sons a meaningful look.
This time the lesson went home. The brothers stopped fighting each other. They would work together as a team and succeed in doing whatever work was given to them.
The four boys had discovered that unity is strength."""

summary=select_sentences(text,10)

#NP = "NP: { <NN>+|<CD> }"
NP = "NP: { <JJ><NN>|<JJ><NNP>|<NN><NN>|<NN>|<NNP><NNP>|<NNP>|<CD> }"
chunker = nltk.RegexpParser(NP)
#print(summary)
for i in summary:
    result = chunker.parse(pos_tag(word_tokenize(i)))
    list_NP=[]
    #print(result)
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        str=""
        for temp in subtree.leaves():
            if(str!=""):#Adding spaces between lines if not first word.
                str+=" "
            str+=temp[0]
        list_NP.append(str)
    #print(list_NP)
    for gaps in list_NP:
        q=i.replace(gaps,"_______")
        print("Original Sentence: {}".format(i))
        print("Question: {}".format(q))
        print("Answer: {}".format(gaps))
        print()

    #result.draw()