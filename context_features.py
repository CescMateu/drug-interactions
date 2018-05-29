# Import basic packages
import nltk
import os
import pandas as pd

# Import self-defined functions
from drug_functions import *
from binary_features_functions import *
from NER_functions import *

def countEntitiesBetweenEntities(sentence_tokenized, ent1, ent2, entities_list):
	'''
	(str, str, str, list of str) -> int
	Description:
	Returns an integer indicating the number of entities between ent1 and ent2. A list of entities is provided in entities_list which includes
	ent1 and ent2. 

	Examples:
	>>> countEntitiesBetweenEntities(['Right', 'now', 'I', 'am', 'trying', 'to', 'work'], 'now', 'trying', ['work', 'now', 'trying', 'am'])
	1
	>>> countEntitiesBetweenEntities(['Oops', ',', 'seems', 'that', 'I', 'took', 'my', 'Ibuprofeno', 'too', 'late'], 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno', 'took', 'I', 'Oops'])
	2
	>>> countEntitiesBetweenEntities(['Oops', ',', 'seems', 'that', 'I', 'took', 'my', 'Ibuprofeno', 'too', 'late'], 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno', 'late', 'Oops'])
	0
	>>> countEntitiesBetweenEntities(['Oops', ',', 'seems', 'that', 'I', 'took', 'my', 'Ibuprofeno', 'too', 'late'], 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno'])
	0
	'''

	# Debugging
	#import pdb; pdb.set_trace()

	# Transform all the inputs into lower case
	#ent1 = ent1.lower()
	#ent2 = ent2.lower()
	#sentence_tokenized = [el.lower() for el in sentence_tokenized]
	#entities_list_aux = [el.lower() for el in entities_list]
	#entities_list = entities_list_aux

	entities_list_local = list(entities_list)
	#print(sentence_tokenized)
	#print(ent1)
	#print(ent2)
	#print(entities_list)

	# Eliminate the reference entities from the list of entities
	entities_list_local.remove(ent1)
	entities_list_local.remove(ent2)

	# If the length of the entities_list is 0, return a 0 (Only two entities were provided)
	if len(entities_list_local) == 0:
	    return(0)

	# Get the range of the sentence in which to iterate
	if sentence_tokenized.index(ent1) <	sentence_tokenized.index(ent2):
	    min_idx = sentence_tokenized.index(ent1)
	    max_idx = sentence_tokenized.index(ent2)
	else:
	    min_idx = sentence_tokenized.index(ent2)
	    max_idx = sentence_tokenized.index(ent1)

	# Iterate over the sentence between ent1 and ent2 looking for other entities
	counter_entities = 0
	for idx in range(min_idx, max_idx):
	    if sentence_tokenized[idx] in entities_list_local:
	        entities_list_local.remove(sentence_tokenized[idx])
	        counter_entities += 1

	return(counter_entities)

def getFirstModalVerb(sentence_tokenized):
	'''

	>>> getFirstModalVerb(['You', 'should', 'not', 'take', 'this', 'medicament', '.'])
	'should'
	>>> getFirstModalVerb(['My', 'house', 'is', 'yours'])
	'none'
	'''

	modal_verbs = ['should', 'must', 'can', 'could', 'may', 'might', 'will', 'would', 'shall']

	for token in sentence_tokenized:
		if token in modal_verbs:
			return(token)
	return('none')


def countModalVerbsBetweenEntities(sentence_tokenized, ent1, ent2):
	'''
	Description:
	Returns an integer indicating the number of modal verbs in the sentence between ent1 and ent2

	>>> countModalVerbsBetweenEntities(['I', 'think', 'you', 'should', 'eat', 'your', 'ice-cream'], 'you', 'ice-cream')
	1
	>>> countModalVerbsBetweenEntities(['I', 'think', 'you', 'should', 'eat', 'your', 'ice-cream'], 'your', 'ice-cream')
	0
	'''
	# Initializate a list with the most common modal verbs
	modal_verbs = ['should', 'must', 'can', 'could', 'may', 'might', 'will', 'would', 'shall']

	# Get the range of the sentence in which to iterate
	if sentence_tokenized.index(ent1) <	sentence_tokenized.index(ent2):
	    min_idx = sentence_tokenized.index(ent1)
	    max_idx = sentence_tokenized.index(ent2)
	else:
	    min_idx = sentence_tokenized.index(ent2)
	    max_idx = sentence_tokenized.index(ent1)

	# Iterate over the list to retrieve the result
	modal_verbs_count = 0
	for modal_verb in modal_verbs:
		if modal_verb in sentence_tokenized[min_idx:max_idx+1]:
			modal_verbs_count += 1

	return(modal_verbs_count)

def sentenceContainsNegation(sentence_tokenized):
	'''

	Tests:
	>>> sentenceContainsNegation(['You', 'should', 'not', 'drink', 'coke'])
	1
	>>> sentenceContainsNegation(["You", "should", "n't", "drink", "coke"])
	1
	>>> sentenceContainsNegation(['You', 'can', 'drink', 'any', 'coke', 'you', 'want'])
	0
	'''
	negations = ["not", "n't"]

	for token in sentence_tokenized:
		if token in negations:
			return(int(True))

	return(int(False))

def keyWordsBetweenEntities(sentence_tokenized, ent1, ent2):
	'''
	
	>>> keyWordsBetweenEntities(['Indicated', 'that', 'PIDIroxyne', 'significantly', 'reduced', 'Neurotoxicity'], 'PIDIroxyne', 'Neurotoxicity')
	1
	>>> keyWordsBetweenEntities(['Indicated', 'that', 'PIDIroxyne', 'significantly', 'reduced', 'Neurotoxicity'], 'that', 'PIDIroxyne')
	0
	>>> keyWordsBetweenEntities(['Indicated', 'that', 'PIDIroxyne', 'significantly', 'did', 'nothing', 'Neurotoxicity'], 'Indicated', 'Neurotoxicity')
	0
	>>> keyWordsBetweenEntities(['Indicated', 'that', 'TNF Blocking Agents', 'significantly', 'did', 'nothing', 'Neurotoxicity'], 'TNF Blocking Agents', 'Neurotoxicity')
	0
	>>> keyWordsBetweenEntities(['Indicated', 'that', 'TNF Blocking Agents', 'significantly', 'increased', 'Neurotoxicity'], 'TNF Blocking Agents', 'Neurotoxicity')
	1
	'''

	# Define the key words that we want to look for
	key_words = '^increase|^reduce|^diminish|^affect|^inhib'

	# Look for the indices of the entities, so we can iterate over the words in the middle
	idx1 = sentence_tokenized.index(ent1)
	idx2 = sentence_tokenized.index(ent2)
	sentence_tokenized = sentence_tokenized[idx1:idx2+1]

	# Iterate over the selected tokens to see if there is a match with the keywords
	for token in sentence_tokenized:
		if re.search(key_words, token):
			return(int(True))

	return(int(False))

def createSimplifiedPOSPath(pos_tags):
	'''

	>>> createSimplifiedPOSPath(['NNP', 'NNA', 'NNV', 'VB', ',', 'VB', 'VB', 'PRA', 'PRE', 'ADJ', 'AVV'])
	'NN-VB-,-VB-PR-AD-AV'
	'''
	
	# Only retain the two first characters of each POS tag
	pos_tags_2char = [tag[0:2] for tag in pos_tags]

	# Attach the first pos_tag to the simplified list
	simpl_pos_tag_list = [pos_tags_2char[0]]

	# Iterate over the given pos_tags list and omit the repetitions
	for tag in pos_tags_2char:
		if tag != simpl_pos_tag_list[-1]:
			simpl_pos_tag_list.append(tag)

	return('-'.join(simpl_pos_tag_list))


def getNgramsBetweenEntities(sentence_tokenized, ent1, ent2, n):

	'''

	>>> getNgramsBetweenEntities(['Hello', 'my', 'name', 'is', 'Cesc'], 'my', 'Cesc', 3)
	['my ', 'y n', ' na', 'nam', 'ame', 'me ', 'e i', ' is', 'is ', 's C', ' Ce', 'Ces', 'esc']

	'''


	# Get the range of the sentence in which to iterate
	if sentence_tokenized.index(ent1) <	sentence_tokenized.index(ent2):
	    min_idx = sentence_tokenized.index(ent1)
	    max_idx = sentence_tokenized.index(ent2)
	else:
	    min_idx = sentence_tokenized.index(ent2)
	    max_idx = sentence_tokenized.index(ent1)


	words_to_convert = ' '.join(sentence_tokenized[min_idx:max_idx+1])

	return(num_ngrams(words_to_convert, n))



def countTokensBetweenEntities(sentence_tokenized, ent1, ent2):
	'''

	Description:
	Returns an integer indicating the number of tokens between entity1 and entity2, independently from the real order in the sentence.
	Entity1 and entity2 are also counted in the integer returned. If the entity is not in the sentence, an exception is raised.

	Examples:
	>>> countTokensBetweenEntities(['My', 'name', 'is', 'Cesc', 'Mateu'], 'My', 'Cesc')
	3
	>>> countTokensBetweenEntities(['My', 'name', 'is', 'Cesc', 'Mateu'], 'Cesc', 'My')
	3
	>>> countTokensBetweenEntities(['I', 'am', 'listening', 'to', 'music'], 'music', 'music')
	0
	>>> countTokensBetweenEntities(['I', 'am', 'listening', 'to', 'music'], 'to', 'music')
	1
	>>> countTokensBetweenEntities(['I', 'am', 'listening', 'to', 'music'], 'am', 'music')
	3
	'''

	if ent1 not in sentence_tokenized:
		raise ValueError(ent1, ' was not found in ', sentence_tokenized)
	if ent2 not in sentence_tokenized:
		raise ValueError(ent2, ' was not found in ', sentence_tokenized)

	return(abs(sentence_tokenized.index(ent1)-sentence_tokenized.index(ent2)))

def numberOfEntities(entities_list):

	entities_list_lower = [el.lower() for el in entities_list]
	return(len(set(entities_list)))

def has2Ent(entities_list):
	return(numberOfEntities(entities_list) == 2)

def has3EntOrMore(entities_list):
	return(numberOfEntities(entities_list) >= 3)

def allEntAfterNeg(sentence_tokenized, entities_list):

	'''
	>>> allEntAfterNeg(['My', 'name', 'is', 'not', 'Cesc'], ['Cesc'])
	1
	>>> allEntAfterNeg(['My', 'name', 'is', 'not', 'Cesc'], ['Cesc', 'is'])
	0
	>>> allEntAfterNeg(['My', 'name', 'is', 'Cesc'], ['Cesc', 'is'])
	0
	'''

	negations = ["not", "n't"]
	n_entities_found = 0

	for token in sentence_tokenized:
		if token in entities_list:
			n_entities_found =+ 1

		if token in negations:
			if n_entities_found > 0:
				return(int(False))
			else:
				return(int(True))

	return(int(False))

def sentenceContainsBut(sentence_tokenized):

	for token in sentence_tokenized:
		if token == 'but':
			return(int(True))

	return(int(False))

def sentenceContainsContrastExp(sentence_tokenized):
	'''

	>>> sentenceContainsContrastExp(['My', 'name', 'is', 'Cesc', 'although', 'my', 'friends', 'call', 'me', 'Kiku'])
	1
	>>> sentenceContainsContrastExp(['We', 'enjoyed', 'our', 'camping', 'holiday', 'in', 'spite', 'of', 'the', 'rain'])
	1
	>>> sentenceContainsContrastExp(['We', 'enjoyed', 'our', 'camping', 'holiday'])
	0
	'''

	contrast_words = ['although', 'though', 'despite']

	for i in range(len(sentence_tokenized)):

		if sentence_tokenized[i] in contrast_words:
			return(int(True))

		if i >= 1:
			if sentence_tokenized[i] == 'spite' and sentence_tokenized[i-1] == 'in':
				return(int(True))
	return(int(False))


def getPOSTagNeighbours(sentence_tokenized, entity, position, pos_tags):
	'''

	>>> getPOSTagNeighbours(['My', 'name', 'is', 'Cesc'], 'is', -1, ['DTE', 'NNT', 'VBP', 'NNT'])
	'NN'
	>>> getPOSTagNeighbours(['My', 'name', 'is', 'Cesc'], 'is', +2, ['DTE', 'NNT', 'VBP', 'NNT'])
	''
	>>> getPOSTagNeighbours(['My', 'name', 'is', 'Cesc'], 'is', +1, ['DTE', 'NNT', 'VBP', 'NNT'])
	'NN'
	'''

	pos_tags_aux = [tag[0:2] for tag in pos_tags]

	idx_ent = sentence_tokenized.index(entity)

	if (idx_ent+position >= 0) and (idx_ent+position <= (len(pos_tags_aux)-1)):
		return(pos_tags_aux[idx_ent+position])
	else:
		return('')







if __name__ == '__main__':
    import doctest
    doctest.testmod()
