import xml.etree.ElementTree as ET # ElementTree Library

def listEntitiesFromXML(file_root_xml):

	''' (xml.root.file) -> list of list of str

	Documentation
    
	Given an XML file root file_root_xml, this function returns a list of list of str with all the different entities inside the XML file.

	Test Examples
	...

	'''

	# Initialise the list accumulator for all the entities
	file_entities = []

	# Iterate over the 'sentence' elements in the file
	for sentence in file_root_xml.iter('sentence'):

		# Get the main sentence
		sentence_text = sentence.get('text')
		sentence_id = sentence.get('id')

		# Iterate over all the items of the sentence
		for item in sentence:

			if item.tag == 'entity':
				entity_name = item.get('text')
				entity_id = item.get('id')
				entity_type = item.get('type')
				entity_charOffset = item.get('charOffset')
				file_entities.append([sentence_id, sentence_text, entity_id, entity_name, entity_charOffset, entity_type])

			else:
				pass

	# Once we have retrieved all the desired information from the 'sentence' element
	# we would like to create a data frame in which each row represents one entity of the sentence.

	# Return the result
	return file_entities



def listDDIFromXML(file_root_xml):
	''' (xml.root.file) -> list of str

	Documentation

	...

	Test Examples
	...

	'''

	# Initialise the list accumulator and the names of the final dataframe
	file_interactions = []

	# Iterate over the 'sentence' elements in the file
	for sentence in file_root_xml.iter('sentence'):
	    # Get the main sentence
	    sentence_text = sentence.get('text')
	    sentence_id = sentence.get('id')

	    # Initialise the list of unique entities in the text
	    n_entities = 0
	    entities_names, entities_ids, entities_types = [], [], []
	    
	    # Initialse the list of the relationships between drugs
	    n_relationships = 0
	    relationships_types = []
	    relationships_element1, relationships_element2 = [], []
	    
	    # Iterate over all the items of the sentence
	    for item in sentence:

	        if item.tag == 'entity':
	            entities_names.append(item.get('text'))
	            entities_ids.append(item.get('id'))
	            entities_types.append(item.get('type'))
	            n_entities += 1
	            
	        elif item.tag == 'pair':
	            relationships_types.append(item.get('ddi'))
	            relationships_element1.append(item.get('e1'))
	            relationships_element2.append(item.get('e2'))
	            n_relationships += 1
	            
	        else:
	            pass
	    
	    # Once we have retrieved all the desired information from the sentence element
	    # we would like to create a data frame in which each row represents one relationship
	    # between two drugs or biologic groups. For each sentence, we will have
	    # as much rows as relationships expressed in the original xml document
	    
	    if n_relationships > 0: # Each relationship will represent one row
	        for i in range(n_relationships):
	            # Get the indexes of each of the elements in the relationship
	            e1_idx, e2_idx = entities_ids.index(relationships_element1[i]), entities_ids.index(relationships_element2[i])
	            
	            # Extract the desired data of each element
	            e1_name, e2_name = entities_names[e1_idx], entities_names[e2_idx]
	            e1_id, e2_id = entities_ids[e1_idx], entities_ids[e2_idx]
	            e1_type, e2_type = entities_types[e1_idx], entities_types[e2_idx]

	            # Which relationship do the 2 elements have?
	            rel_type = relationships_types[i]

	            # Put the data into a list of lists
	            row = [sentence_id, sentence_text, e1_id, e1_name, e1_type, e2_id, e2_name, e2_type, rel_type]
	            file_interactions.append(row)

	# Return the result
	return file_interactions
                