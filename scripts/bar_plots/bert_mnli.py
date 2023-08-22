from collections import Counter, defaultdict
import math

def accuracy_per_model():
	with open(f'../../results/all_mnli_predictions.txt', 'r', encoding='utf8') as f:
		gold_labels = [line.strip().split(',')[-1] for line in f.readlines()]

	with open(f'../../results/all_mnli_development_set_predictions.txt', 'r', encoding='utf8') as f:
		scores = []
		for idx, (line, label) in enumerate(zip(f.readlines(), gold_labels)):
			line = line.strip().split(',')
			preds = line[:-1]

			if len(scores) == 0:
				scores = [1 if p == label else 0 for p in preds]
			else:
				scores = [score + c for score, c in zip(scores, [1 if p == label else 0 for p in preds])]

		scores = [round(score / (idx + 1), 3) for score in scores]
		return {k: round(v / 100, 2) for k, v in zip(Counter(scores).keys(), Counter(scores).values())}


def correct_per_instance():
	counter = Counter()
	with open(f'../../results/all_mnli_predictions.txt', 'r', encoding='utf8') as f:
		gold_labels = [line.strip().split(',')[-1] for line in f.readlines()]

	with open(f'../../results/all_mnli_development_set_predictions.txt', 'r', encoding='utf8') as f:
		for idx, (line, label) in enumerate(zip(f.readlines(), gold_labels)):
			line = line.strip().split(',')
			preds = line[:-1]
			correct = sum((1 if p == label else 0 for p in preds))
			for c in range(correct + 1):
				counter[c] += 1

	all_counts = {round(k / len(preds), 2): round(v / (idx + 1), 2) for k, v in zip(counter.keys(), counter.values())}
	grouped_counts = {}
	for key in all_counts:
		if key == round(math.ceil(key / 0.05) * 0.05, 2):
			grouped_counts[key] = all_counts[key]

	return grouped_counts

cpi = correct_per_instance()

output = ''
for k, v in zip(cpi.keys(), cpi.values()):
	output += f'BERT,{k},{v}\n'

with open('../../results/bert_mnli_accs.csv', 'w') as f:
	f.write(output)
