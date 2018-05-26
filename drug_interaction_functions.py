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


def countTokensBetweenEntities(sentence, ent1, ent2):
	'''

	Description:
	Returns an integer indicating the number of tokens between entity1 and entity2, independently from the real order in the sentence.
	Entity1 and entity2 are also counted in the integer returned. If the entity is not in the sentence, an exception is raised.

	Examples:
	>>> countTokensBetweenEntities('My name is Cesc Mateu', 'My', 'Cesc')
	4
	>>> countTokensBetweenEntities('My name is Cesc Mateu', 'Cesc', 'My')
	4
	>>> countTokensBetweenEntities('I am listening to music', 'music', 'music')
	1
	>>> countTokensBetweenEntities('I am listening to music', 'to', 'music')
	2
	>>> countTokensBetweenEntities('I am listening to music', 'I am', 'music')
	4
	'''
	# Transform all the inputs into lower case
	ent1 = ent1.lower()
	ent2 = ent2.lower()
	sentence = sentence.lower()

	# Tokenize the sentence except for the entities
	sentence_token = tokenizeExceptEntities(sentence, [ent1, ent2])

	return(abs(sentence_token.index(ent1)-sentence_token.index(ent2)) + 1)


def countEntitiesBetweenEntities(sentence, ent1, ent2, entities_list):
	'''
	(str, str, str, list of str) -> int
	Description:
	Returns an integer indicating the number of entities between ent1 and ent2. A list of entities is provided in entities_list which includes
	ent1 and ent2. 

	Examples:
	>>> countEntitiesBetweenEntities('Right now I am trying to work', 'now', 'trying', ['work', 'now', 'trying', 'am'])
	1
	>>> countEntitiesBetweenEntities('Oops, seems that I took my Ibuprofeno too late', 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno', 'took', 'I', 'Oops'])
	2
	>>> countEntitiesBetweenEntities('Oops, seems that I took my Ibuprofeno too late', 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno', 'late', 'Oops'])
	0
	>>> countEntitiesBetweenEntities('Oops, seems that I took my Ibuprofeno too late', 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno'])
	0
	'''

	# Debugging
	#import pdb; pdb.set_trace()

	# Transform all the inputs into lower case
	ent1 = ent1.lower()
	ent2 = ent2.lower()
	sentence = sentence.lower()
	entities_list_aux = [el.lower() for el in entities_list]
	entities_list = entities_list_aux

	# Tokenize the sentence without tokenizing the drug entities 
	sentence_tokenized = tokenizeExceptEntities(sentence, entities_list)

	# Eliminate the reference entities from the list of entities
	entities_list.remove(ent1)
	entities_list.remove(ent2)

	# If the length of the entities_list is 0, return a 0 (Only two entities were provided)
	if len(entities_list) == 0:
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
	    if sentence_tokenized[idx] in entities_list:
	        entities_list.remove(sentence_tokenized[idx])
	        counter_entities += 1

	return(counter_entities)


def countModalVerbsBetweenEntities(sentence, ent1, ent2):
	'''
	Description:
	Returns an integer indicating the number of modal verbs in the sentence between ent1 and ent2

	>>> countModalVerbsBetweenEntities('I think you should eat your ice-cream', 'you', 'ice-cream')
	1
	>>> countModalVerbsBetweenEntities('I think you should eat your ice-cream', 'your', 'ice-cream')
	0
	'''
	# Initializate a list with the most common modal verbs
	modal_verbs = ['should', 'must', 'can', 'could', 'may', 'might', 'will', 'would', 'shall']

	# Transform all the inputs into lower case
	ent1 = ent1.lower()
	ent2 = ent2.lower()
	sentence = sentence.lower()

	# Tokenize the sentence without tokenizing the drug entities
	sentence_tokenized = tokenizeExceptEntities(sentence, [ent1, ent2])

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

def createOrtographicFeatures(drugs_df):
	'''
	Description:
	Given a dataset containing two columns with the names of a pair of entities, the sentence that contains those entities, a list
	with all the entities contained in that sentence and the resulting interaction of the initial entities, 
	this function returns a dataframe in which many different features have been computed
	'''

	# Setting off a very annoying warning
	pd.options.mode.chained_assignment = None  # default='warn'

	for ent_idx in [1, 2]:

		drugs_df['ent%d_contains_numbers' % ent_idx] = drugs_df.apply(
			lambda row: hasNumbers(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_has_uppercase' % ent_idx] = drugs_df.apply(
			lambda row: hasUpperCase(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_all_uppercase' % ent_idx] = drugs_df.apply(
			lambda row: allUpperCase(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_initial_capital' % ent_idx] = drugs_df.apply(
			lambda row: hasInitialCapital(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_contains_slash' % ent_idx] = drugs_df.apply(
			lambda row: containsSlash(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_contains_dash' % ent_idx] = drugs_df.apply(
			lambda row: containsDash(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_n_tokens' % ent_idx] = drugs_df.apply(
			lambda row: countTokensEntity(
				entity = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_contains_punctuation' % ent_idx] = drugs_df.apply(
			lambda row: punctuation(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_init_digit' % ent_idx] = drugs_df.apply(
			lambda row: initDigit(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_single_digit' % ent_idx] = drugs_df.apply(
			lambda row: singleDigit(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_contains_roman' % ent_idx] = drugs_df.apply(
			lambda row: roman(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_end_punctuation' % ent_idx] = drugs_df.apply(
			lambda row: endPunctuation(
				string = row['e%d_name' % ent_idx]),
			axis = 1)

		drugs_df['ent%d_caps_mix' % ent_idx] = drugs_df.apply(
			lambda row: capsMix(
				string = row['e%d_name' % ent_idx]),
			axis = 1)


	return(drugs_df)


def createContextFeatures(drugs_df):
	'''
	Description:
	Given a dataset containing two columns with the names of a pair of entities, the sentence that contains those entities, a list
	with all the entities contained in that sentence and the resulting interaction of the initial entities, 
	this function returns a dataframe in which many different features have been computed
	'''

	# Setting off a very annoying warning
	pd.options.mode.chained_assignment = None  # default='warn'

	drugs_df['n_modal_verbs_bw_entities'] = drugs_df.apply(
		lambda row: countModalVerbsBetweenEntities(
			sentence=row['sentence_text'],
			ent1=row['e1_name'],
			ent2=row['e2_name']),
		axis=1)

	drugs_df['n_tokens_bw_entities'] = drugs_df.apply(
		lambda row: countTokensBetweenEntities(
			sentence=row['sentence_text'],
			ent1=row['e1_name'],
			ent2=row['e2_name']),
		axis = 1)

	drugs_df['n_entities_bw_entities'] = drugs_df.apply(
		lambda row: countEntitiesBetweenEntities(
			sentence=row['sentence_text'],
			ent1=row['e1_name'],
			ent2=row['e2_name'],
			entities_list = row['list_entities']),
		axis = 1)

	drugs_df['POS_path_bw_entities'] = drugs_df.apply(
		lambda row: createPOSpath(
			sentence=row['sentence_text'],
			ent1=row['e1_name'],
			ent2=row['e2_name'],
			simplified = True),
		axis = 1)


	return(drugs_df)


def createPOSpath(sentence, ent1, ent2, simplified = False):
	'''
	Description:
	Path between a pair of drugs: Path between two main drugs in the parse tree is another feature in our system. 
	Because syntactic paths are in general a sparse feature, we reduced the sparsity by collapsing identical adjacent non-terminal labels. 
	E.g., NP-S-VP-VP-NP is converted to NP-S-VP-NP. This technique decreased the number of paths by 24.8%.

	Examples:
	>>> createPOSpath('I like to take Ibuprofenos together with aspirins', 'Ibuprofenos', 'aspirins', simplified = True)
	'RB-IN'
	>>> createPOSpath('I like to take Ibuprofenos Fluimucils together with aspirins', 'I', 'aspirins', simplified = False)
	''
	>>> createPOSpath('I like to take Ibuprofenos Fluimucils together with aspirins', 'I', 'aspirins', simplified = True)
	''
	'''

	# Tokenize the sentence
	#tok_sentence = tokenizeExceptEntities(sentence = sentence, entities_list = [ent1, ent2])
	tok_sentence = nltk.word_tokenize(sentence)
	pos_tok_sentence = nltk.pos_tag(tok_sentence)

	ent1 = nltk.word_tokenize(ent1)
	ent2 = nltk.word_tokenize(ent2)

	# Look for the indices of the pair of drugs inside the sentence
	ent1_idx = tok_sentence.index(ent1[0])
	ent2_idx = tok_sentence.index(ent2[0])

	# Extract the POS tags between the two entities
	if ent1_idx < ent2_idx:
		pos_tags = pos_tok_sentence[ent1_idx+1:ent2_idx]
	else:
		pos_tags = pos_tok_sentence[ent2_idx+1:ent1_idx]

	# Retain only the tags
	pos_tags = [tag[1] for tag in pos_tags]

	# Retain only the first two characters of each POS tag
	pos_tags = [tag[0:2] for tag in pos_tags]


	# Export the result in the desired format (simplified argument)
	res = ''
	if simplified == False:
		for idx, tag in enumerate(pos_tags):
			if idx == len(pos_tags) - 1: # Just append the tag if it's the last tag of the list (without the '-')
				res = res + tag
			else:
				res = res + tag + '-'

		return(res)

	elif simplified == True:

		if len(pos_tags) == 0:
			return('')

		prev_tag = pos_tags[0]
		simplified_res = [prev_tag]

		for tag in pos_tags[1:]:
			if tag != prev_tag:
				simplified_res.append(tag)
				prev_tag = tag

		return('-'.join(simplified_res))

	




def containsPrefix(string, prefix):
	'''
	Description:
	Checks if a string starts with a particular prefix

	>>> containsPrefix('ibuprofeno', 'ibu')
	1
	>>> containsPrefix('Ibuprofeno', 'ibu')
	1
	>>> containsPrefix('TNF blocking ibuprofenos', 'ibu')
	1
	>>> containsPrefix('Ibuprofeno', 'hepta')
	0
	>>> containsPrefix('TNF blocking ibuprofenos', 'hola')
	0
	'''

	string = string.lower()
	prefix = prefix.lower()

	if countTokensEntity(string) == 1:
		# If the entity only is formed by one word
		return(int(string.startswith(prefix)))
	else:
		# If the entity is formed by more than one word
		string_sep = nltk.word_tokenize(string)
		for word in string_sep:
			if word.startswith(prefix):
				return(int(True))
		return(int(False))

def containsSuffix(string, suffix):
	'''
	Description:
	Checks if a string starts with a particular suffix

	>>> containsSuffix('ibuprofeno', 'eno')
	1
	>>> containsSuffix('IbuprofenO', 'eno')
	1
	>>> containsSuffix('TNF blocking ibuprofeno', 'eno')
	1
	>>> containsSuffix('Ibuprofeno', 'hepta')
	0
	>>> containsSuffix('TNF blocking ibuprofenos', 'hola')
	0
	'''

	string = string.lower()
	suffix = suffix.lower()

	if countTokensEntity(string) == 1:
		# If the entity only is formed by one word
		return(string.endswith(suffix))
	else:
		# If the entity is formed by more than one word
		string_sep = nltk.word_tokenize(string)
		for word in string_sep:
			if word.endswith(suffix):
				return(int(True))
		return(int(False))


def createMorphologicFeatures(drugs_df):

	# Setting off a very annoying warning
	pd.options.mode.chained_assignment = None  # default='warn'

	prefixes = ['alk', 'meth', 'eth', 'prop', 'but', 'pent', 'hex', 'hept', 'oct', 'non', 'dec', 'undec', 'dodec', 'eifcos', 'di', 'tri', 'tetra', 'penta', 'hexa', 'hepta']
	sufixes = ['ane', 'ene', 'yne', 'yl', 'ol', 'al', 'oic', 'one', 'ate', 'amine', 'amide']

	for ent_idx in [1, 2]:
		for prefix in prefixes:
			drugs_df['ent{ent_idx}_contains_prefix_{prefix}'.format(ent_idx = ent_idx, prefix = prefix)] = drugs_df.apply(
				lambda row: containsPrefix(
					string = row['e%d_name' % ent_idx],
					prefix = prefix),
				axis = 1)

		for suffix in sufixes:
			drugs_df['ent{ent_idx}_contains_suffix_{suffix}'.format(ent_idx = ent_idx, suffix = suffix)] = drugs_df.apply(
				lambda row: containsSuffix(
					string = row['e%d_name' % ent_idx],
					suffix = suffix),
				axis = 1)

	return(drugs_df)





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

def keyWordsBetweenEntities(sentence, ent1, ent2):
	'''
	
	>>> keyWordsBetweenEntities('Indicated that PIDIroxyne significantly reduced Neurotoxicity', 'PIDIroxyne', 'Neurotoxicity')
	1
	>>> keyWordsBetweenEntities('Indicated that PIDIroxyne significantly reduced Neurotoxicity', 'that', 'PIDIroxyne')
	0
	>>> keyWordsBetweenEntities('Indicated that PIDIroxyne significantly did nothing Neurotoxicity', 'Indicated', 'Neurotoxicity')
	0
	>>> keyWordsBetweenEntities('Indicated that TNF Blocking Agents significantly did nothing Neurotoxicity', 'TNF Blocking Agents', 'Neurotoxicity')
	0
	>>> keyWordsBetweenEntities('Indicated that TNF Blocking Agents significantly increased Neurotoxicity', 'TNF Blocking Agents', 'Neurotoxicity')
	1
	'''

	key_words = '^increase|^reduce|^diminish|^affect'

	tokens = tokenizeExceptEntities(sentence, [ent1, ent2])

	idx1 = tokens.index(ent1)
	idx2 = tokens.index(ent2)

	tokens = tokens[idx1:idx2+1]

	for token in tokens:
		if re.search(key_words, token):
			return(int(True))

	return(int(False))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    