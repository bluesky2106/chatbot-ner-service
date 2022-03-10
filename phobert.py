import pickle
import torch
import numpy as np
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer

class PhoBERT(object):
	def __init__(self, label_path='resources/tags.pkl', model_path='resources/phobert') -> None:
		with open(label_path, 'rb') as f:
			self.__tags = pickle.load(f)

		self.__tag2idx = {t: i for i, t in enumerate(self.__tags)}
		self.__idx2tag = {i: t for i, t in enumerate(self.__tags)}

		self.__model = torch.load(model_path)
		self.__model.eval()

		self.__annotator = VnCoreNLP(address="http://127.0.0.1", port=8000)
		self.__tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

	def predict_sentence(self, sentence):
		segmented_words = self.__annotator.tokenize(sentence)
		segmented_text = [' '.join(word) for word in segmented_words]
		segmented_text = ' '.join(segmented_text)

		input_ids = torch.tensor([self.__tokenizer.encode(segmented_text)])

		with torch.no_grad():
			output = self.__model(input_ids)
		label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

		tokens = self.__tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
		new_tokens, new_tags = [], []
		for token, label_idx in zip(tokens, label_indices[0]):
			if token == "<s>" or token == "</s>":
				continue
			if token.startswith("##"):
				new_tokens[-1] = new_tokens[-1] + token[2:]
			else:
				new_tags.append(self.__tags[label_idx])
				new_tokens.append(token)

		new_tokens, new_tags = self.__convert_subwords_to_text(new_tokens, new_tags)
		return self.__generate_content_label(new_tokens, new_tags)

	def __convert_subwords_to_text(self, tokens, tags):
		idx = 0
		tks, ts = [], []
		while idx < len(tags):
			current_tag = tags[idx]
			current_token = tokens[idx]
			updated = True
			while current_token.endswith("@@"):
				idx += 1
				if idx < len(tags):
					if tokens[idx] != "," and tokens[idx] != ".":
						current_token += " " + tokens[idx]
						current_token = current_token.replace("@@ ", "")
						if current_tag == "O":
							current_tag = tags[idx]
					else:
						updated = False
						current_token = current_token.replace("@@", "")
						break
				else:
					current_token = current_token.replace("@@", "")
			tks.append(current_token)
			ts.append(current_tag)
			if updated:
				idx += 1

		return tks, ts

	def __generate_content_label(self, tokens, tags):
		contents, labels = [], []
		idx = 0
		while idx < len(tags):
			ts = []
			current_tag = tags[idx]
			current_token = tokens[idx]
			
			if current_tag == "O":
				current_label = current_tag
				ts.append(current_token)
				labels.append(current_label)

				idx += 1
				while idx < len(tags):
					next_tag = tags[idx]
					if next_tag != "O":
						break
					next_token = tokens[idx]
					ts.append(next_token)
					idx += 1
				
				content = ' '.join(ts)
				contents.append(content.replace("_", " "))
				continue

			if current_tag.startswith('B-') or current_tag.startswith('I-'):
				current_label = current_tag[2:]
				ts.append(current_token)
				labels.append(current_label)

				idx += 1
				while idx < len(tags):
					next_tag = tags[idx]
					next_label = next_tag[2:]
					if next_label != current_label:
						break
					next_token = tokens[idx]
					ts.append(next_token)
					idx += 1
				
				content = ' '.join(ts)
				contents.append(content.replace("_", " "))
				continue

		return contents, labels