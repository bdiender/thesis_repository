from collections import Counter, defaultdict
import math

def accuracy_per_model():
	with open(f'../../results/all_mnli_predictions.txt', 'r', encoding='utf8') as f:
		scores = []
		for idx, line in enumerate(f.readlines()):
			line = line.strip().split(',')
			preds = line[:-1]
			label = line[-1]
			
			if len(scores) == 0:
				scores = [1 if p == label else 0 for p in preds]
			else:
				scores = [score + c for score, c in zip(scores, [1 if p == label else 0 for p in preds])]

		scores = [round(score / (idx + 1), 3) for score in scores]
		return {k: round(v / 29, 2) for k, v in zip(Counter(scores).keys(), Counter(scores).values())}


def correct_per_instance():
	counter = Counter()
	with open(f'../../results/all_mnli_predictions.txt', 'r', encoding='utf8') as f:
		for idx, line in enumerate(f.readlines()):
			line = line.strip().split(',')
			preds = line[:-1]
			label = line[-1]
			correct = sum((1 if p == label else 0 for p in preds))
			for c in range(correct, len(preds) + 1):
				counter[c] += 1

	all_counts = {round(k / len(preds), 2): round(v / (idx + 1), 2) for k, v in zip(counter.keys(), counter.values())}
	ceilings = {}
	for key in all_counts:
		if round(math.ceil(key / 0.05) * 0.05, 2) not in ceilings:
			ceilings[round(math.ceil(key / 0.05) * 0.05, 2)] = key
		elif key > ceilings[round(math.ceil(key / 0.05) * 0.05, 2)]:
			ceilings[round(math.ceil(key / 0.05) * 0.05, 2)] = key
	
	return {key: all_counts[ceilings[key]] for key in ceilings}

cpi = correct_per_instance()

output = ''
for k, v in zip(cpi.keys(), cpi.values()):
	output += f'RoBERTa,{k},{v}\n'

with open('../../results/roberta_mnli_accs.csv', 'w') as f:
	f.write(output)

