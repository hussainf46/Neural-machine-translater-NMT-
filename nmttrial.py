# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:42:33 2019

@author: Dell
"""

import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
#from pickle import load
#from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
#from pickle import load
#from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from pickle import load
#from numpy import array
from numpy import argmax
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.text import text_to_word_sequence
 
#part 1 cleaning the dataset 


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs
 
# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))

	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)
 
# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')
# spot check
for i in range(100):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
	
	
#part 2 :- splitting the datset into testset and train set 
	
	
# load a clean dataset
def load_clean_sentences(filename):
	#fil = open(filename,rb)
	return load(open(filename, 'rb'))#this is used to load file first argument is the file name in the open portion 
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))# this is used to save the datset or a anything into a file the first argument is the valye to be
	#stored and tghe second argument is the filename with write permission
	print('Saved: %s' % filename)
 
# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')
 
# reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :2]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:9000], dataset[9000:]
# save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')
	

# part 3 :- training the model 


# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# fit a tokenizer
#fit_on_texts Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on
#word frequency. So if you give it something like, "The cat sat on the mat." It will create a dictionary 
#s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value. 0 is reserved for padding. So lower integer means more frequent word (often the first few are stop words because they appear a lot)
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
	
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)
	
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])#this is used to give each word in datset a integer vvalue 
eng_vocab_size = len(eng_tokenizer.word_index) + 1#the word_index has integer fo each and every word in the above dataset and plus is because the 0 is eeserved for the padding bit
#the above will give the number of different word in our dataset 
eng_length = max_length(dataset[:, 0])#this gives us the len of the line with maximum number of words 
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))


# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y


# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model
 
# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())	
#plot_model(model, to_file='model.png', show_shapes=True)
	
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
	
#part:- 4 evaluating the model 


# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)
 
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
 
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources,raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		#print (source)
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target,raw_src = raw_dataset[i]
		    
		if i < 10:
			#print(raw_src)
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		
		actual.append([j.split() for j in raw_target])
		predicted.append(translation.split())		
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

 
# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
 
# load model
model = load_model('model.h5')
# test on some training sequences
print('train')
#x = input("enter the string :-")
evaluate_model(model, eng_tokenizer, trainX, train )
# test on some test sequences
print('test')
evaluate_model(model, eng_tokenizer, testX ,test)




#part 5 entering data and testing 
#from keras.preprocessing.text import text_to_word_sequence
#z="tom ist ein spion"
#z=text_to_word_sequence(z)
#print(zt)
#ge_tokenizer = create_tokenizer(z)
#ge_vocab_size = len(ge_tokenizer.word_index) + 1
#ge_length = max_length(z)

#eg_tokenizer = create_tokenizer(z)
#eg_vocab_size = len(eg_tokenizer.word_index) + 1
#eg_length = max_length(z)







#z=get_ans(input(enter the string:-))
#z=text_to_word_sequence(z)
def get_ans(z):
	al=[]
	review = re.sub('[^a-zA-Z]',' ', z)
	al.append(review)
	zg=encode_sequences(ger_tokenizer,ger_length,al)
	translation = predict_sequence(model,eng_tokenizer,zg)
	print('src=[%s],predicted=[%s]' % (z,  translation))
	
#x=input("enter the string:-")
for i in range(5):
	z=get_ans(input("enter the string:-"))



#tokenizer= Tokenizer()
#z="tom ist ein spion"
#al=[]
#review = re.sub('[^a-zA-Z]',' ', z)
#print(review)
#al.append(review)
#print(al)
#tokenizer.fit_on_texts(z)
#zg=tokenizer.texts_to_sequences(z)


#x = input("enter the string :-")
	
	
	