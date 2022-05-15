import os
import torch
import numpy as np

from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer, BertForQuestionAnswering, BertTokenizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_crf import CRFModel

from constants import *

PADDING_TAG = "PAD"
SENTENCE_LENGTH = 250

def build_bilstm_crf_model(n_labels, n_vocab, input_length, embdding_dim, lstm_units, dropout):
	input = Input(shape=(input_length,))
	word_emb = Embedding(input_dim=n_vocab, output_dim=embdding_dim, input_length=input_length)(input)
	bilstm = Bidirectional(LSTM(units=lstm_units, return_sequences=True,
							recurrent_dropout=dropout))(word_emb)
	dense = Dense(n_labels, activation="relu")(bilstm)
	base = Model(inputs=input, outputs=dense)
	model = CRFModel(base, n_labels)

	return model

class NERModel(object):
	def __init__(self) -> None:
		with open('resources/tags.txt', 'r') as f:
			self.__tags = [line.rstrip('\n') for line in f]

		vncorenlp_svc_host = os.getenv('vncorenlp_svc_host')
		if not vncorenlp_svc_host:
			vncorenlp_svc_host = "http://127.0.0.1"

		vncorenlp_svc_port = os.getenv('vncorenlp_svc_port')
		if not vncorenlp_svc_port:
			vncorenlp_svc_port = "8000"

		self.__annotator = VnCoreNLP(address=vncorenlp_svc_host, port=int(vncorenlp_svc_port))
		self.__base_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
		self.__large_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large", use_fast=False)

		self.__phobert_base_model = torch.load(RESOURCE_PHOBERT_BASE, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		self.__phobert_base_model.eval()

		self.__phobert_large_model = torch.load(RESOURCE_PHOBERT_LARGE, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		self.__phobert_large_model.eval()

		self.__bilstm_model = keras.models.load_model(RESOURCE_PHOBERT_BILSTM)

		self.__bilstm_crf_model = build_bilstm_crf_model(len(self.__tags), 64000, SENTENCE_LENGTH, 1024, 128, 0.1)
		self.__bilstm_crf_model.load_weights(RESOURCE_PHOBERT_BILSTM_CRF)
		self.__bilstm_crf_model.compile(optimizer="adam", metrics=['acc'])

	def __split_array(self, arr, max_len):
		if len(arr) <= max_len:
			return [arr.copy()]
		idx = max_len
		# 4 = . ; 5 = ,
		while arr[idx] != 4 and arr[idx] != 5 and idx > 0:
			idx -= 1
		if idx == 0:
			idx = max_len
		results = [arr[:idx].copy()]
		results.extend(self.__split_array(arr[idx:].copy(), max_len))
		return results

	def predict_sentence(self, sentence, model_name):
		segmented_words = self.__annotator.tokenize(sentence)
		segmented_words = [word for sublist in segmented_words for word in sublist]
		segmented_text = " ".join(segmented_words)

		if model_name == MODEL_PHOBERT_LARGE:
			x = self.__large_tokenizer.encode(segmented_text, add_special_tokens=True)
		else:
			x = self.__base_tokenizer.encode(segmented_text, add_special_tokens=True)
		xs = self.__split_array(x, SENTENCE_LENGTH)

		new_tokens, new_tags = [], []
		for x in xs:
			if model_name == MODEL_PHOBERT_BASE:
				input_ids = torch.tensor([x])
				with torch.no_grad():
					y = self.__phobert_base_model(input_ids)
				label_indices = np.argmax(y[0].to('cpu').numpy(), axis=2)
				label_indices = label_indices[0]
				tokens = self.__base_tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
			elif model_name == MODEL_PHOBERT_LARGE:
				input_ids = torch.tensor([x])
				with torch.no_grad():
					y = self.__phobert_large_model(input_ids)
				label_indices = np.argmax(y[0].to('cpu').numpy(), axis=2)
				label_indices = label_indices[0]
				tokens = self.__large_tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
			elif model_name == MODEL_BILSTM:
				length = len(x)
				if length > SENTENCE_LENGTH:
					length = SENTENCE_LENGTH
				x = pad_sequences([x], maxlen=SENTENCE_LENGTH, dtype="long", value=0.0, truncating="post", padding="post")
				y = self.__bilstm_model.predict(x)
				label_indices = np.argmax(y[0], axis=-1)
				label_indices = label_indices[:length]
				input_ids = x[0][:length]
				tokens = self.__base_tokenizer.convert_ids_to_tokens(input_ids)
			elif model_name == MODEL_BILSTM_CRF:
				length = len(x)
				if length > SENTENCE_LENGTH:
					length = SENTENCE_LENGTH
				x = pad_sequences([x], maxlen=SENTENCE_LENGTH, dtype="long", value=0.0, truncating="post", padding="post")
				y = self.__bilstm_crf_model.predict(x)
				label_indices = y[0][0]
				label_indices = label_indices[:length]
				input_ids = x[0][:length]
				tokens = self.__base_tokenizer.convert_ids_to_tokens(input_ids)

			for token, label_idx in zip(tokens, label_indices):
				if token == "<s>" or token == "</s>":
					continue
				new_tags.append(self.__tags[label_idx])
				new_tokens.append(token)

		new_tokens, new_tags = self.__convert_subwords_to_text(new_tokens, new_tags)
		new_tokens = self.__fix_unknown_tokens(new_tokens, segmented_text)
		return self.__generate_content_label(new_tokens, new_tags)
		
	def __convert_subwords_to_text(self, tokens, tags):
		idx = 0
		tks, ts = [], []
		while idx < len(tags):
			current_tag = tags[idx]
			if current_tag == PADDING_TAG:
				current_tag = "O"

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
				content = content.replace(" _ ", " ")
				content = content.replace("_", " ")
				contents.append(content)
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
				content = content.replace(" _ ", " ")
				content = content.replace("_", " ")
				contents.append(content)
				continue

		return contents, labels

	def __fix_unknown_tokens(self, tokens, segmented_text):
		current_pos = 0
		for idx, token in enumerate(tokens):
			if token == "<unk>" and idx < len(tokens)-1:
				next_token = tokens[idx+1]
				pos = segmented_text.find(next_token, current_pos)
				if pos >= 0:
					tokens[idx] = segmented_text[current_pos:pos-1]
			current_pos += len(tokens[idx]) + 1
		return tokens

class QuestionAnsweringModel(object):
	def __init__(self) -> None:
		self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
		self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

	def question_answer(self, question, text):
		# print("\nQuestion:\n{}".format(question.capitalize()))
		
		#tokenize question and text as a pair
		input_ids = self.tokenizer.encode(question, text)
		
		#string version of tokenized ids
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
		
		#segment IDs
		#first occurence of [SEP] token
		sep_idx = input_ids.index(self.tokenizer.sep_token_id)
		#number of tokens in segment A (question)
		num_seg_a = sep_idx+1
		#number of tokens in segment B (text)
		num_seg_b = len(input_ids) - num_seg_a
		
		#list of 0s and 1s for segment embeddings
		segment_ids = [0]*num_seg_a + [1]*num_seg_b
		# assert len(segment_ids) == len(input_ids)
		
		#model output using input_ids and segment_ids
		output = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
		
		#reconstructing the answer
		answer_start = torch.argmax(output.start_logits)
		answer_end = torch.argmax(output.end_logits)
		if answer_end < answer_start:
			answer = "Unable to find the answer to your question."
		else:
			answer = tokens[answer_start]
			for i in range(answer_start+1, answer_end+1):
				if tokens[i][0:2] == "##":
					answer += tokens[i][2:]
				else:
					answer += " " + tokens[i]
					
		if answer.startswith("[CLS]"):
			answer = "Unable to find the answer to your question."
		
		# print("\nPredicted answer:\n{}".format(answer.capitalize()))

		return answer