import json
from flask import Flask, request, jsonify
from phobert import PhoBERT

app = Flask(__name__)
app.config["DEBUG"] = True

bert = PhoBERT()

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
	
	if model == "BERT":
		contents, labels = bert.predict_sentence(text)
		response = []
		for content, label in zip(contents, labels):
			print("{}\t{}".format(content, label))
			response.append({"content": content, "label": label})
		return jsonify(response)
	
	return "NER"

app.run(host="0.0.0.0")