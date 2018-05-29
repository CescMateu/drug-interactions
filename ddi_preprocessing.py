
import re

def preprocessing(dataset):
	'''
	>>> preprocessing([('hola  tal not que a',['que', 'hola']),('segona sentence must not be',['entity1','entity2']),('segona baby must not be',['entity1','entity2'])])
	[('hola  tal not que a', ['que', 'hola'])]
	>>> preprocessing([('hola que tal not recommended',['que']),('segona sentence must not be',['entity1','entity2'])])
	[]
	>>> preprocessing([("hey man, you shouldn't do that!", ['that']), ('I would like to buy some groceries for tonight', ['groceries', 'tonight']), ('In the book it says that is not recommended to mix alcohol with marihuana', ['alcohol', 'marihuana'])])
	[]
	'''

	re_1 = r'\bnot recommended|should not be|must not be\b'
	re_2 = r'\bno|n\'t|not\b'
	new_dataset=[]
	for text in dataset:
		
		# clear one-or-zero DDI sentences | clear sentences containing "not recommended”, “should not be” or “must not be" | clear sentences not containing “no”, “n’t” or “not” 
		# clear sentence if  no target entity mention appears in the sentence after “no”, “n’t” or “not”
		if len(text[1])>=2 and not re.search(re_1,text[0]) and re.search(re_2,text[0]):

			# this is for the fourth clearing criteria 
			m = re.search(re_2,text[0])
			neg_cue_matched = m.group(0)
			initial_pos = text[0].index(neg_cue_matched)
			new_sentence = text[0][initial_pos:]
			for entity in text[1]:
				if entity in new_sentence:
					new_dataset.append(text)
					continue
		 

		
	return new_dataset

if __name__ == '__main__':
    import doctest
    doctest.testmod()