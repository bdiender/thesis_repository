from collections import Counter, defaultdict
import math

def correct_per_instance():
	counter_heuristic = defaultdict(Counter)
	counter_subcase = defaultdict(Counter)
	with open('../../results/hans_subcases.txt', 'r', encoding='utf8') as f:
		subcases = [line.split(',') for line in f.readlines()]

	with open('../../results/all_hans_predictions.txt', 'r', encoding='utf8') as f:
		gold_labels = [line.strip().split(',')[-1] for line in f.readlines()]

	with open(f'../../results/mccoy_hans_predictions.txt', 'r', encoding='utf8') as f:
		for line, (heuristic, subcase), label in zip(f.readlines(), subcases, gold_labels):
			preds = line.strip().split(',')


	with open(f'../../results/mccoy_hans_predictions.txt', 'r', encoding='utf8') as f:
		for line, (heuristic, subcase), label in zip(f.readlines(), subcases, gold_labels):
			preds = line.strip().split(',')
			correct = sum((1 if pred == label else 0 for pred in preds))
			correct /= 100
			correct = round(math.ceil(correct / 0.1) * 0.1, 2)
			counter_heuristic[heuristic][correct] += 1
			counter_subcase[subcase][correct] += 1

	return counter_heuristic, counter_subcase


def accuracy_per_model():
	scorecard_heuristic = defaultdict(list)
	scorecard_subcase = defaultdict(list)

	with open('../../results/hans_subcases.txt', 'r', encoding='utf8') as f:
		subcases = [line.strip().split(',') for line in f.readlines()]

	with open('../../results/all_hans_predictions.txt', 'r', encoding='utf8') as f:
		gold_labels = [line.strip().split(',')[-1] for line in f.readlines()]
	
	with open(f'../../results/mccoy_hans_predictions.txt', 'r', encoding='utf8') as f:
		for line, (h, s), label in zip(f.readlines(), subcases, gold_labels):
			line = line.strip().split(',')
			preds = line[:-1]

			if len(scorecard_heuristic[h]) == 0:
				scorecard_heuristic[h] = [1 if p == label else 0 for p in preds]
			else:
				scorecard_heuristic[h] = [score + correct for score, correct in zip(scorecard_heuristic[h], [1 if p == label else 0 for p in preds])]

			if len(scorecard_subcase[s]) == 0:
				scorecard_subcase[s] = [1 if p == label else 0 for p in preds]
			else:
				scorecard_subcase[s] = [score + correct for score, correct in zip(scorecard_subcase[s], [1 if p == label else 0 for p in preds])]

	for scorecard in (scorecard_heuristic, scorecard_subcase):
	 	for key in scorecard:
	 		accs = []
	 		for ncor in scorecard[key]:
	 			ncor /= len(subcases)
	 			ncor *= 6
	 			ncor = round(math.ceil(ncor / 0.025) * 0.025, 3)
	 			accs.append(ncor)
	 		scorecard[key] = Counter(accs)
	 		for acc in scorecard[key]:
	 			scorecard[key][acc] /= 100


	return scorecard_heuristic, scorecard_subcase


counter_heuristic, counter_subcase = accuracy_per_model()
for h in counter_heuristic:
	for key, value in zip(counter_heuristic[h].keys(), counter_heuristic[h].values()):
		heuristic_dict = {
			"c": "Constituent",
			"e": "Consistent",
			"l": "Lexical overlap",
			"n": "Inconsistent",
			"s": "Subsequence"
			}
		print(f'BERT,{heuristic_dict[h[0]]},{key},{value},{heuristic_dict[h[1]]}')
