from argparse import ArgumentParser
import pickle

import pandas as pd
import numpy as np
import huggingface_hub

import dspy
from dspy.teleprompt import BootstrapFewShot

from dspy.evaluate import Evaluate
from tqdm import tqdm

from data_utils import make_dataset, parse_args

TOKEN = "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"

huggingface_hub.login(token=TOKEN)



if __name__ == '__main__':
    arguments = parse_args()

    llama3 = dspy.HFModel(model = arguments.model)
    dspy.settings.configure(lm=llama3)
    classify = dspy.Predict('question, choices -> answer', temperature=0.1)


    train_dataset = make_dataset(arguments.train_data, arguments.train_labels)
    

    dev_dataset = make_dataset(arguments.dev_data, arguments.dev_labels)
    print("LOADED DATASET")
    answers = {}

    for example in tqdm(dev_dataset):
        id_ = example.id_
        response = classify(question=example.question, choices=example.choices)
        answers[id_] = response._store

    with open(arguments.output, 'wb') as f:
        pickle.dump(answers, f)