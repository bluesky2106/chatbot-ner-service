import json
from flask import Flask, request, jsonify
from model import NERModel

MAX_SENTENCE_LENGTH = 200

app = Flask(__name__)
app.config["DEBUG"] = False

ner_model = NERModel()

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
	if model != "BERT" and model != "BiLSTM" and model != "BiLSTM+CRF":
		return jsonify({'error': 'only accept BERT or BiLSTM or BiLSTM+CRF models'})
	
	ss = text.split()
	if len(ss) > MAX_SENTENCE_LENGTH:
		texts = split_text(text)
	else:
		texts = [text]

	response = []
	for txt in texts:
		contents, labels = ner_model.predict_sentence(txt, model)
		for content, label in zip(contents, labels):
			# print("{}\t{}".format(content, label))
			response.append({"content": content, "label": label})

	return jsonify(response)

def split_text(text):
	texts = []
	if len(text) <= MAX_SENTENCE_LENGTH:
		texts.append(text)
		return texts

	idx = MAX_SENTENCE_LENGTH-1
	while text[idx] != "." and idx > 0:
		idx -= 1
	if idx == 0:
		idx = MAX_SENTENCE_LENGTH-1

	txt1 = text[:idx+1]
	txt2 = text[idx+1:]
	texts.append(txt1)
	texts.extend(split_text(txt2))
	return texts

if __name__ == '__main__':
  	app.run(host='0.0.0.0')