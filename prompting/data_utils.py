import json

import dspy
import pandas as pd
import numpy as np

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='google/gemma-1.1-2b-it')
    parser.add_argument('--temperature', type=float, default=0.1)
    # parser.add_argument('--train_data', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/train.jsonl')
    # parser.add_argument('--train_labels', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/train-labels.lst')
    # parser.add_argument('--dev_data', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/dev.jsonl')
    parser.add_argument('--dev_data', type=str, default='/d/hpc/home/in7357/data/ANLI_R1/dev.jsonl')
    # parser.add_argument('--dev_labels', type=str, default='/d/hpc/home/in7357/data/physicaliqa-train-dev/dev-labels.lst')
    parser.add_argument('--output', type=str, default='/d/hpc/home/in7357/ul-fri-nlp-course-project-randomly_generated/model_outputs/anli/GEMMA_baseline-TEST_100.pkl')

    return parser.parse_args()


def make_dataset_anli(questions_path):
    dataset = [] 
    with open(questions_path) as f:
            question_data = [json.loads(line) for line in f]

    df = pd.DataFrame(question_data)

    for id_, context, hypothesis, label in df[['uid', 'context', 'hypothesis', 'label']].values:
    
        dataset.append(dspy.Example(id_=id_, context=context, hypothesis=hypothesis, 
                                    selection=label).with_inputs('question', 'choices'))

    return dataset



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


