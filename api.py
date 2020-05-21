from flask import Flask, jsonify, request, abort, g
from SentenceSelector import select_sentences
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import nltk
from textblob import TextBlob
from flask_cors import CORS

app = Flask(__name__) # Flask object created
CORS(app) # CORS (Cross-Origin Resource Sharing) is a standard for accessing web resources on different domains.

@app.route('/summary', methods = ['POST'])
def generateSummary():
	print(request)
	data = request.get_json()
	text = data['Text']
	required_lines = int(data['Lines'])
	summary = select_sentences(text, required_lines)
	Summary = []
	for it in summary:
		Summary.append(it)
	return {
			"Summary": Summary
		}	

@app.route('/answer_list', methods = ['POST'])
def generateQA():
	data = request.get_json()
	summary = []
	final_qa_list = []
	for it in data['Summary']:
		summary.append(it)
	#NP = "NP: { <NN>+|<CD> }"
	NP = "NP: { <JJ><NN>|<JJ><NNP>|<NN><NN>|<NN>|<NNP><NNP>|<NNP>|<CD> }"
	chunker = nltk.RegexpParser(NP)
	#print(summary)
	count = 0
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
	        cur_qa = {}
	        cur_qa['Original_Sentence'] = i
	        cur_qa['Question'] = q
	        cur_qa['Answer'] = gaps
	        # print("Original Sentence: {}".format(i))
	        # print("Question: {}".format(q))
	        # print("Answer: {}".format(gaps))
	        # print()
	        final_qa_list.append(cur_qa)
	        count += 1
	    #result.draw()
	return {
			"Number_of_question": count,
			"QAs": final_qa_list
		}


@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.debug = True
    app.run()