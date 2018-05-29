# Import basic packages
import nltk
import os
import pandas as pd

# Import self-defined functions
from drug_functions import *
from binary_features_functions import *

def tokenizeExceptEntities(sentence, entities_list):
    ''' (str, list of str) -> list of str
    Description: 
    This function tokenizes a given sentence except for those entities that are in the sentence and are in the list. This
    just applies for those entities that are multiword, for instance 'TNF blocking agents'

    Examples:
    >>> tokenizeExceptEntities('I like to take TNF blocking agents for breakfast morning', ['TNF blocking agents', 'breakfast'])
    ['I', 'like', 'to', 'take', 'TNF blocking agents', 'for', 'breakfast', 'morning']
    >>> tokenizeExceptEntities('Hey, be careful when drinking calcium enriched uranium.', ['calcium enriched uranium'])
    ['Hey', ',', 'be', 'careful', 'when', 'drinking', 'calcium enriched uranium', '.']
    '''

    # Find the beginning and the end of each entity in the sentence
    entities_length = []
    for entity in entities_list:
        initial_char = sentence.find(entity)
        final_char = initial_char + len(entity)
        entities_length.append([initial_char, final_char])
    
    # Break the sentence in small chunks and just tokenize the non-entities parts
    initial_idx = 0
    tokenized_sentence = []
    for idx, entity_length in enumerate(entities_length):
        final_idx = entity_length[0]
        sentence_chunk = sentence[initial_idx:final_idx]
        tokenized_sentence.append(nltk.word_tokenize(sentence_chunk))
        tokenized_sentence.append(entities_list[idx])
        initial_idx = entity_length[1] + 1

    final_sentence_chunk = sentence[(initial_idx-1):]
    tokenized_sentence.append(nltk.word_tokenize(final_sentence_chunk))
        
    # Transform the list of sublists into a single list
    flat_list = []
    for subelement in tokenized_sentence:
        if (isinstance(subelement, list)): 
            if len(subelement) > 1:
                for item in subelement:
                    flat_list.append(item)
            elif len(subelement) == 1:
                flat_list.append(subelement[0])
        else:
            flat_list.append(subelement)
    
    return(flat_list)


def computeConfusionMatrix(true, pred):

	if(len(true) != len(pred)):
		stop('Provided vectors do not have the same length')

	# Initialize the different counters
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0

	for t, p in zip(true, pred):
		if t == 1 and p == 1:
			true_pos += 1
		elif t == 1 and p == 0:
			false_neg += 1
		elif t == 0 and p == 1:
			false_pos += 1
		elif t == 0 and p == 0:
			true_neg += 1


	return([true_pos, true_neg, false_pos, false_neg])

def computePrecision(true, pred):
	'''
	Precision (P) is defined as the number of true positives (T_p) over the number of true positives plus the number of false positives (F_p).
	P = \frac{T_p}{T_p+F_p}
	'''

	conf_matrix = computeConfusionMatrix(true,pred)

	if conf_matrix[0] == 0:
		return(0)
	
	return conf_matrix[0]/(conf_matrix[0] + conf_matrix[2])


def computeRecall(true, pred):
	'''
	Recall (R) is defined as the number of true positives (T_p) over the number of true positives plus the number of false negatives (F_n).
	R = \frac{T_p}{T_p + F_n}
	'''

	conf_matrix = computeConfusionMatrix(true,pred)

	if conf_matrix[0] == 0:
		return(0)

	return(conf_matrix[0]/(conf_matrix[0] + conf_matrix[3]))

def computeF1(true, pred):
	'''
	These quantities are also related to the (F_1) score, which is defined as the harmonic mean of precision and recall.
	F1 = 2\frac{P \times R}{P+R}
	'''

	prec = computePrecision(true, pred)
	rec = computeRecall(true, pred)

	if prec+rec == 0:
		return(0)

	return(2*(prec*rec)/(prec+rec))


def sentenceContainsNegation(sentence):
	'''

	Tests:
	>>> sentenceContainsNegation("You should not drink coke")
	1
	>>> sentenceContainsNegation("You shouldn't drink coke")
	1
	>>> sentenceContainsNegation("You can drink any coke you want")
	0
	'''

	tokens = nltk.word_tokenize(sentence)

	negations = ["not", "n't"]

	for token in tokens:
		if token in negations:
			return(int(True))

	return(int(False))

def frequency(entity1,entity2,distrFrequencies):
    num1 = distrFrequencies[entity1]
    num2 = distrFrequencies[entity2]
    den = num1+num2
    if den==0: return 0
    else:
        return (2*num1*num2)/den


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    