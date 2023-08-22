from collections import defaultdict
from itertools import combinations
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np

with open('../../results/all_mnli_development_set_predictions.txt', 'r') as f:
	gold_labels = [line.strip().split(',')[-1] for line in f.readlines()]

with open('../../results/all_mnli_development_set_predictions.txt', 'r') as f:
    errors_per_model = defaultdict(set)

    for idl, (line, label) in enumerate(zip(f.readlines(), gold_labels)):
        line = line.strip().split(',')
        preds = line[:-1]

        for idm, p in enumerate(preds):
            if p != label:
                errors_per_model[idm].add(idl)

overlap_ratios = {}
for model1, model2 in combinations(errors_per_model.keys(), 2):
    intersection = len(errors_per_model[model1].intersection(errors_per_model[model2]))
    union = len(errors_per_model[model1].union(errors_per_model[model2]))
    overlap_ratios[(model1, model2)] = intersection / union

num_instances = len(gold_labels)
num_models = len(errors_per_model)
fleiss_matrix = np.zeros((num_instances, num_models))

for model, errors in errors_per_model.items():
    for instance in range(num_instances):
        if instance in errors:
            fleiss_matrix[instance, model] = 1
        else:
            fleiss_matrix[instance, model] = 2

fleiss_counts = np.zeros((num_instances, 2))
for instance in range(num_instances):
    category1_count = np.count_nonzero(fleiss_matrix[instance, :] == 1)
    category2_count = np.count_nonzero(fleiss_matrix[instance, :] == 2)
    fleiss_counts[instance, 0] = category1_count
    fleiss_counts[instance, 1] = category2_count

fleiss_kappa_score = fleiss_kappa(fleiss_counts)
print(fleiss_kappa_score)

output = ''
for pair in overlap_ratios:
    output += f'BERT;{round(overlap_ratios[pair], 3)}\n'

with open('../../results/bert_mnli_eor.csv', 'w') as f:
    f.write(output)
