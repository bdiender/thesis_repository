from collections import defaultdict
from itertools import combinations
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np

with open('../../results/hans_subcases.txt', 'r') as f:
    subcases = ['_'.join(line.strip().split(',')) for line in f.readlines()]

errors_per_model = {subcase: defaultdict(set) for subcase in set(subcases)}

with open('../../results/mccoy_hans_predictions.txt', 'r') as f:
    gold_labels = [line.strip().split(',')[-1] for line in f.readlines()]

with open('../../results/mccoy_hans_predictions.txt', 'r') as f:
    for idl, (line, subcase, label) in enumerate(zip(f.readlines(), subcases, gold_labels)):
        preds = line.strip().split(',')[:-1]
        for idm, p in enumerate(preds):
            if p != label:
                errors_per_model[subcase][idm].add(idl)

categories = {'ce', 'cn', 'le', 'ln', 'se', 'sn'}

per_category = {category: defaultdict(set) for category in set(categories)}
for subcase in set(subcases):
    for model in errors_per_model[subcase]:
        per_category[subcase[:2]][model] = per_category[subcase[:2]][model].union(errors_per_model[subcase][model])

overlap_ratios = {category: {} for category in categories}

for category in set(categories):
    for model1, model2 in combinations(per_category[category].keys(), 2):
        intersection = len(per_category[category][model1].intersection(per_category[category][model2]))
        union = len(per_category[category][model1].union(per_category[category][model2]))
        overlap_ratios[category][(model1, model2)] = intersection / union

output = ''
for cat in overlap_ratios:
    for pair in overlap_ratios[cat]:
        heuristic = {'c': 'Constituent', 'l': 'Lexical overlap', 's': 'Subsequence'}[cat[0]]
        consistent = {'e': 'Consistent', 'n': 'Inconsistent'}[cat[1]]
        output += f'RoBERTa;{heuristic};{round(overlap_ratios[cat][pair], 3)};{consistent}\n'

with open('../../results/bert_hans_eor.csv', 'w') as f:
    f.write(output)

categories = {'ce', 'cn', 'le', 'ln', 'se', 'sn'}
num_instances = len(gold_labels)

fleiss_matrices = {category: np.zeros((num_instances, len(per_category[category]))) for category in categories}

for category in categories:
    for model, errors in per_category[category].items():
        for instance in errors:
            fleiss_matrices[category][instance, model] = 1

    for instance in range(num_instances):
        for model in range(len(per_category[category])):
            if fleiss_matrices[category][instance, model] != 1:
                fleiss_matrices[category][instance, model] = 2


fleiss_counts = {category: np.zeros((num_instances, 2)) for category in categories}
for category in categories:
    for instance in range(num_instances):
        category1_count = np.count_nonzero(fleiss_matrices[category][instance, :] == 1)
        category2_count = np.count_nonzero(fleiss_matrices[category][instance, :] == 2)
        fleiss_counts[category][instance, 0] = category1_count
        fleiss_counts[category][instance, 1] = category2_count

fleiss_kappa_scores = {}
for category in categories:
    fleiss_kappa_scores[category] = fleiss_kappa(fleiss_counts[category])

for category, kappa_score in fleiss_kappa_scores.items():
    heuristic = {'c': 'Constituent', 'l': 'Lexical overlap', 's': 'Subsequence'}[category[0]]
    consistent = {'e': 'Consistent', 'n': 'Inconsistent'}[category[1]]
    print(f'BERT;{heuristic};{consistent};{round(kappa_score, 3)}')
