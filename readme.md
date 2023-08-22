# Random Seed Influence on Language Model Generalizability

This repository contains the scripts used in my thesis.

## Repository Structure

- `./results/`: Contains the files that contain all predictions, as well as files that contain the accuracies and error overlap ratios
- `./results/hans`: Contains the results of the evaluation of the RoBERTa models on HANS
- `./results/mnli`: Contains the results of the evaluation of the RoBERTa models on MNLI
- `./scripts/`: Contains the main scripts for running experiments.
- `./scripts/analyses`: Contains the scripts used for the IAA analyses and the linguistic analysis reported in Section 4.3.1.
- `./scripts/bar_plots`: Contains the scripts used for obtaining the values shown in the bar plots in the thesis.

## Usage

To run `main.py`, the script that trains and evaluates the RoBERTa instances:

```
python ./scripts/main.py 
    -S <first_seed> 
    -C <number_of_models>
    -T <use_toydata_true_or_false>
    -E <number_of_epochs>
    -L <save_model_true_or_false>
```

This script was executed on DAS-5 and stores the output files in a location different from where the other scripts in this repository expect them. The csv-files containing the results should be placed in the `./results/hans` or `./results/mnli` folders.

The script `collect_predictions.py` converts the csv-files to txt-files that contain all predictions for all instances of RoBERTa. It stores them as `all_hans_predictions.txt` and `all_mnli_predictions` in the `./results` folder. The results reported by McCoy, Min, & Linzen (2020) can be taken from [their repository](https://github.com/tommccoy1/hans/tree/master/berts_of_a_feather), their file containing the predictions on HANS should be renamed to `mccoy_hans_predictions.txt`. Both of these files should be placed in the `./results` folder.

Values of the accuracies are extracted from the files with the four scripts in the `./scripts/bar_plots` folder. Each of these scripts calculates the share of models per accuracy score for each model type and each dataset separately, and saves them to csv files that are placed in the `./results` folder

The scripts used to measure the agreement are found in the `./scripts/analyses` folder. Similarly to the `./scripts/bar_plots` folder, there are four scripts that calculate the IAA values used separately for both types of models and both datasets and stores them in csv-files in the `./results` folder. Although the linguistic analysis found no correlation, the script is included in this folder, too, as `semantic_similarity.py`. This script prints its outputs and does not store them to a file.
