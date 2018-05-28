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
