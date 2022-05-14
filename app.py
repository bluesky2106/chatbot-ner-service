import json
from flask import Flask, request, jsonify
from model import NERModel, QuestionAnsweringModel
from constants import *

app = Flask(__name__)
app.config["DEBUG"] = False

ner_model = NERModel()
question_answering_model = QuestionAnsweringModel()

@app.route('/', methods=['GET'])
def home():
    return "Homepage"

@app.route('/api/v1/ner', methods=['POST'])
def extract_entities():
	record = json.loads(request.data)
	model = record.get('model')
	text = record.get('text')

	if model is None or text is None:
		return jsonify({'error': 'wrong request body format'})
	if model != MODEL_PHOBERT_BASE and model != MODEL_PHOBERT_LARGE and \
		model != MODEL_BILSTM and model != MODEL_BILSTM_CRF:
		return jsonify({'error': 'only accept {} / {} / {} / {} models'.format(MODEL_PHOBERT_BASE, MODEL_PHOBERT_LARGE, MODEL_BILSTM, MODEL_BILSTM_CRF)})
	
	response = []
	contents, labels = ner_model.predict_sentence(text, model)
	for content, label in zip(contents, labels):
		# print("{}\t{}".format(content, label))
		response.append({"content": content, "label": label})

	return jsonify(response)

@app.route('/api/v1/question-answering', methods=['POST'])
def extract_answer():
	record = json.loads(request.data)
	question = record.get('question')
	passage = record.get('passage')

	if question is None or passage is None:
		return jsonify({'error': 'wrong request body format'})
	
	answer = question_answering_model.question_answer(question, passage)
	response = {
		"question": question,
		"answer": answer
	}
	return jsonify(response)

if __name__ == '__main__':
  	app.run(host='0.0.0.0', port=8080)