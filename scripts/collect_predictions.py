# The goal of this script is to take all csv evaluation files and create txt files that contain all predictions.
import os
import pandas as pd


def csv_to_lists(directory):
	'''
	Read csv files in the directory and collect them as a list of lists:
	Outer list contains as many examples as there are lines in the test set,
	inner list contains all predictions, as well as the gold label as the last
	item in the list. 
	'''
	preds_per_file = []
	gold_labels = []
	subcases = []
	if directory[-5:-1] == 'hans':
		for file in os.listdir(directory):
			df = pd.read_csv(directory + file, delimiter=';')
			if len(gold_labels) == 0:
				gold_labels = [l for l in df.label]
			if len(subcases) == 0:
				subcases = [f'{s[:2]},{s[3:]}' for s in df.subcase]

			predictions = [p for p in df.prediction]
			preds_per_file.append(predictions)

	elif directory[-5:-1] == 'mnli':
		for file in os.listdir(directory):
			with open(directory + file, 'r', encoding='utf8') as f:
				lines = f.readlines()
				if len(gold_labels) == 0:
					gold_labels = [line[-2] for line in lines][1:]
				predictions = [line[-4] for line in lines][1:]
				preds_per_file.append(predictions)

	preds_per_file.append(gold_labels)
	if not os.path.exists('../results/hans_subcases.txt'):
		with open('../results/hans_subcases.txt', 'w') as f:
			f.write('\n'.join(subcases))

	return list(zip(*preds_per_file))



for dataset in ('hans', 'mnli'):
	results_dir = f'../results/{dataset}/'
	result = csv_to_lists(results_dir)
	output = []
	for line in result:
		if dataset == 'hans':
			line = ['entailment' if i == 0 else 'contradiction' for i in line]
		else:
			line = ['entailment' if i == '0' else 'neutral' if i == '1' else 'contradiction' for i in line]
		output.append(','.join(line))
	
	with open(f'../results/all_{dataset}_predictions.txt', 'w') as f:
		f.write('\n'.join(output))