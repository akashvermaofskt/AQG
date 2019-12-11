from SentenceSelector import select_sentences
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import nltk
from textblob import TextBlob

text="""World War II thus far, has been the deadliest and bloodiest war to date. More than 38 million people died by the end of the war, many of them innocent civilians. It was also the most destructive war in our current history. The fighting raged on in many parts of the world, with the brunt of it being in Europe and Japan. More than 50 nations took part in this war, which changed the world forever. For Americans, World War II had a clear-cut purpose; they were fighting to defeat tyranny. Most of Europe had been conquered by Nazi Germany, which was under the evil control of Adolf Hitler. The war in Europe began with Germany’s unprecedented invasion of Poland in 1939. It seemed that wherever the Nazi army went, they came down with a vengeance on the Jews of that area. They also went after anyone that didnt fit in to their idea of the Master Race, Aryans."""

summary=select_sentences(text,10)

NP = "NP: { <NNP>+|<CD> }"
#NP = "NP: { <JJ><NN>|<JJ><NNP>|<NN><NN>|<NN>|<NNP><NNP>|<NNP>|<CD> }"
chunker = nltk.RegexpParser(NP)

for i in summary:
    #print(i)
    result = chunker.parse(pos_tag(word_tokenize(i)))
    list_NP=[]
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        str=""
        for temp in subtree.leaves():
            if(str!=""):
                str+=" "
            str+=temp[0]
        list_NP.append(str)

    #print(list_NP)
    
    for gaps in list_NP:
        q=i.replace(gaps,"_______")
        print("Question: {}".format(q))
        print("Answer: {}".format(gaps))
        print()

    #result.draw()