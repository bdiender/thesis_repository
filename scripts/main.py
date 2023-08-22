from argparse import ArgumentParser
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import datetime
import torch

def tokenize(batch):
    return tokenizer(batch['premise'], batch['hypothesis'], truncation=True, padding=True)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-S', '--seed',
                    dest='seed',
                    help='The random seed to start with.',
                    default=1, type=int)
    parser.add_argument('-C', '--count',
                    dest='count',
                    help='The number of models trained in this run.',
                    default=1, type=int)
    parser.add_argument('-T', '--toydata',
                    dest='toydata',
                    help='Whether the toy data is used or not.',
                    default=False, type=bool)
    parser.add_argument('-E', '--epochs',
                    dest='epochs',
                    help='The number of epochs for training.',
                    default=3, type=int)
    parser.add_argument('-L', '--save_model',
                    dest='save_model',
                    help='Whether to save the model.',
                    default=False, type=bool)

    args = parser.parse_args()
    device = torch.device("cuda")

    # Training:
    if not torch.cuda.is_available():
        raise RuntimeError
    else:
        print(f"There are {torch.cuda.device_count()} GPUs available")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    ## Load in data
    train_data = load_dataset('glue', 'mnli')['train']
    val_data = load_dataset('glue', 'mnli')['validation_matched']

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

    model.to(device)

    ## Pre-processing
    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)

    ### Create toy data for training
    toy_data = train_data.select(range(2))

    for s in range(args.seed, args.seed + args.count):
        training_args = TrainingArguments(
                do_train=True,
                output_dir=f'./output_s{s}e{args.epochs}/',
                seed=s,
                save_strategy="no",
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=2,
                learning_rate=2e-5,
                fp16=True,
                num_train_epochs=args.epochs# ,
                # report_to="wandb"
        )

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                tokenizer=tokenizer,
                compute_metrics=load_metric('glue', 'mnli'))

        if args.toydata:
            trainer.train_dataset = toy_data

        trainer.train()
        model_name = f'roberta_mnli_s{s}e{args.epochs}.pt'

        if args.toydata:
            model_name = f'small_{model_name}'

        if args.save_model:
            torch.save(model.state_dict(), f'./output_s{s}e{args.epochs}/{model_name}')

        # Evaluation:
        for dataset in ('mnli', 'hans'):
            if dataset == 'hans':
                val_data = load_dataset('hans')['validation']
            else:
                val_data = load_dataset('glue', 'mnli')['validation_matched']

            val_data_tokenized = val_data.map(tokenize, batched=True)
            val_data_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            if dataset == 'hans':
                output = 'premise;hypothesis;prediction;label;heuristic;subcase\n'
                for idx, row in enumerate(val_data_tokenized):
                    with torch.no_grad():
                        input_ids = row['input_ids'].unsqueeze(0).to(device)
                        attention_mask = row['attention_mask'].unsqueeze(0).to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        prediction = torch.argmax(outputs.logits, dim=-1).item()

                        # Convert to binary classification used by HANS
                        prediction = 1 if prediction != 0 else 0

                        premise = val_data[idx]['premise']
                        hypothesis = val_data[idx]['hypothesis']
                        heuristic = val_data[idx]['heuristic']
                        subcase = val_data[idx]['subcase']
                        output += f'{premise};{hypothesis};{prediction};{row["label"]};{heuristic};{subcase}\n'

            else:
                output = output = 'premise;hypothesis;prediction;label\n'
                for idx, row in enumerate(val_data_tokenized):
                    with torch.no_grad():
                        input_ids = row['input_ids'].unsqueeze(0).to(device)
                        attention_mask = row['attention_mask'].unsqueeze(0).to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                        prediction = torch.argmax(outputs.logits, dim=-1).item()
                        premise = val_data[idx]['premise']
                        hypothesis = val_data[idx]['hypothesis']
                        output += f'{premise};{hypothesis};{prediction};{row["label"]}\n'

            with open(f'{dataset}_s{s}e{args.epochs}.csv', 'w') as f:
                f.write(output)

