from datasets import load_dataset
import numpy as np
import spacy


dataset = load_dataset('glue', 'mnli')['validation_matched']
ph_pairs = [(d['premise'], d['hypothesis']) for d in dataset]

nlp = spacy.load('en_core_web_lg')
similarity_scores = []
remove_stops = lambda x: nlp(' '.join(str(t) for t in nlp(x) if not t.is_stop))

for idx, (p, h) in enumerate(ph_pairs):
	p, h = remove_stops(p), remove_stops(h)
	similarity_scores.append(p.similarity(h))

# Read in the RoBERTa predictions on MNLI
gold_labels = []
with open('../../results/all_mnli_predictions.txt', 'r') as f:
	roberta_scores = []
	for idx, line in enumerate(f.readlines()):
		line = line.strip().split(',')
		preds = line[:-1]
		gold_labels.append(line[-1])

		correct_predictions = [p == line[-1] for p in preds]
		roberta_scores.append(sum(correct_predictions) / len(correct_predictions))

# Read in the BERT predictions on MNLI
with open('../../results/all_mnli_development_set_predictions.txt', 'r') as f:
	bert_scores = []
	for idx, (line, label) in enumerate(zip(f.readlines(), gold_labels)):
		line = line.strip().split(',')
		correct_predictions = [p == label for p in line]
		bert_scores.append(sum(correct_predictions) / len(correct_predictions))

similarity_scores_r = [s for s, r in zip(similarity_scores, roberta_scores) if r != 1]
similarity_scores_b = [s for s, b in zip(similarity_scores, bert_scores) if b != 1]

roberta_scores = [r for r in roberta_scores if r != 1]
bert_scores = [b for b in bert_scores if b != 1]

print(len(similarity_scores_r))
print(len(roberta_scores))
print()
print(len(similarity_scores_b))
print(len(bert_scores))

corr_matrix_r = np.corrcoef(roberta_scores, similarity_scores_r)
corr_matrix_b = np.corrcoef(bert_scores, similarity_scores_b)

corr_r = corr_matrix_r[0, 1]
corr_b = corr_matrix_b[0, 1]
Rsq_r = corr_r ** 2
Rsq_b = corr_b ** 2

print(Rsq_r)
print(Rsq_b)
print()
print()
print()
print(corr_matrix_r)
print()
print(corr_matrix_b)
print()
for b, r, q in zip(bert_scores, roberta_scores, similarity_scores):
	print(f'{b};{r};{q}')