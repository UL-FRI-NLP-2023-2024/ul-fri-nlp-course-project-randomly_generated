import json

import dspy
import pandas as pd
import numpy as np

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='google/gemma-1.1-2b-it')
<<<<<<< HEAD
    # parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--train_data', type=str, default='/home/ivan/FRI/2023-2024/sem2/nlp/data/physicaliqa-train-dev/train.jsonl')
    parser.add_argument('--train_labels', type=str, default='/home/ivan/FRI/2023-2024/sem2/nlp/data/physicaliqa-train-dev/train-labels.lst')
    parser.add_argument('--dev_data', type=str, default='/home/ivan/FRI/2023-2024/sem2/nlp/data/physicaliqa-train-dev/dev.jsonl')
    parser.add_argument('--dev_labels', type=str, default='/home/ivan/FRI/2023-2024/sem2/nlp/data/physicaliqa-train-dev/dev-labels.lst')
    parser.add_argument('--output', type=str, default='/home/ivan/FRI/2023-2024/sem2/nlp/data/GEMMA_responses_baseline_NO_REASONING.pick')
=======
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--train_data', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/train.jsonl')
    parser.add_argument('--train_labels', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/train-labels.lst')
    parser.add_argument('--dev_data', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/dev.jsonl')
    parser.add_argument('--dev_labels', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/dev-labels.lst')
    parser.add_argument('--output', type=str, default='/d/hpc/home/in7357/ul-fri-nlp-course-project-randomly_generated/model_outputs/GEMMA_responses_COT-imporved-prompts.pick')
>>>>>>> 9442862 (Add latest code)
    return parser.parse_args()

def make_dataset(questions_path, labels_path):
    dataset = []
    
    with open(labels_path) as f:
        labels = [int(line.strip()) for line in f]
    
    with open(questions_path) as f:
            question_data = [json.loads(line) for line in f]

    df = pd.DataFrame(question_data)
    df['label'] = labels

    for id_, question, answ0, answ1, label in df.values:
        if label == 0:
            answer = answ0
        else:
            answer = answ1

        choices = f"--0:{answ0} \n --1:{answ1}"
    
        dataset.append(dspy.Example(id_=id_, question=question, choices=choices, 
                                    selection=label).with_inputs('question', 'choices'))

    return dataset

def accuracy_metric(example, pred, trace):
    print("Answer: ", example.answer)
    print("Pred:", pred.answer)

    return example.answer == pred.answer

def calculate_accuracy(program, devset):
    scores = []
    for x in devset:
        pred = program(**x.inputs())
        score = accuracy_metric(x, pred, None)
        scores.append(score)
    
    return np.mean(np.array(scores))


