from collections import Counter, defaultdict
import math


def accuracy_per_model():
	scorecard_heuristic = defaultdict(list)
	scorecard_subcase = defaultdict(list)

	with open('../../results/hans_subcases.txt', 'r', encoding='utf8') as f:
		subcases = [line.strip().split(',') for line in f.readlines()]
	
	with open(f'../../results/all_hans_predictions.txt', 'r', encoding='utf8') as f:
		for line, (h, s) in zip(f.readlines(), subcases):
			line = line.strip().split(',')
			preds = line[:-1]
			label = line[-1]

			if len(scorecard_heuristic[h]) == 0:
				scorecard_heuristic[h] = [1 if p == label else 0 for p in preds]
			else:
				scorecard_heuristic[h] = [score + correct for score, correct in zip(scorecard_heuristic[h], [1 if p == label else 0 for p in preds])]

			if h == 'ln':
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
	 			scorecard[key][acc] /= 30


	return scorecard_heuristic, scorecard_subcase


def correct_per_instance():
	counter_heuristic = defaultdict(Counter)
	counter_subcase = defaultdict(Counter)
	with open('../../results/hans_subcases.txt', 'r', encoding='utf8') as f:
		subcases = [line.split(',') for line in f.readlines()]

	with open(f'../../results/all_hans_predictions.txt', 'r', encoding='utf8') as f:
		for line, (heuristic, subcase) in zip(f.readlines(), subcases):
			line = line.strip().split(',')
			preds = line[:-1]
			label = line[-1]
			correct = sum((1 if p == label else 0 for p in preds))
			correct /= 30
			correct = round(math.ceil(correct / 0.1) * 0.1, 2)
			counter_heuristic[heuristic][correct] += 1
			counter_subcase[subcase][correct] += 1

	return counter_heuristic, counter_subcase

counter_heuristic, counter_subcase = accuracy_per_model()
for s in counter_subcase:
	for key, value in zip(counter_subcase[s].keys(), counter_subcase[s].values()):
		print(f'RoBERTa,{s},{key},{round(value, 3)}')


# for h in counter_heuristic:
# 	for key, value in zip(counter_heuristic[h].keys(), counter_heuristic[h].values()):
# 		heuristic_dict = {
# 			"c": "Constituent",
# 			"e": "Consistent",
# 			"l": "Lexical overlap",
# 			"n": "Inconsistent",
# 			"s": "Subsequence"
# 			}
# 		print(f'RoBERTa,{heuristic_dict[h[0]]},{key},{value},{heuristic_dict[h[1]]}')


