import json
from flask import Flask, request, jsonify
from model import NERModel, QuestionAnsweringModel
# from model import NERModel
from constants import *

app = Flask(__name__)
app.config["DEBUG"] = False

ner_model = NERModel()
question_answering_model = QuestionAnsweringModel()

valid_models = [
	MODEL_PHOBERT_BASE,
	MODEL_PHOBERT_LARGE,
	MODEL_BILSTM,
	MODEL_BILSTM_CRF,
	MODEL_PHOBERT_BASE_BILSTM,
	MODEL_PHOBERT_BASE_BILSTM_CRF,
	MODEL_PHOBERT_LARGE_BILSTM,
	MODEL_PHOBERT_LARGE_BILSTM_CRF
]

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
	if model not in valid_models:
		return jsonify({'error': 'only accept {} models'.format(valid_models)})
	
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
	language = record.get('language')

	if question is None or passage is None or language is None:
		return jsonify({'error': 'wrong request body format'})
	if language != LANGUAGE_EN and language != LANGUAGE_VI:
		return jsonify({'error': 'only accept en or vi language'})
	
	answer = question_answering_model.question_answer(question, passage, language)
	response = {
		"answer": answer
	}
	return jsonify(response)

if __name__ == '__main__':
  	app.run(host='0.0.0.0', port=8080)