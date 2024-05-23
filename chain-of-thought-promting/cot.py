import pickle
import json

from tqdm import tqdm

import pandas as pd
import numpy as np
import huggingface_hub

import dspy

from data_utils import make_dataset, parse_args


<<<<<<< HEAD
TOKEN = "TOKEN from https://huggingface.co/settings/tokens"
=======
>>>>>>> 9442862 (Add latest code)
TOKEN = "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"

huggingface_hub.login(token=TOKEN)



class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought('question, choices -> reasoning, Number of the selection [0 or 1]', 
                                        temperature=0.1)
    
    def forward(self, question, choices):
        return self.prog(question=question, choices=choices)


if __name__ == '__main__':
    arguments = parse_args()

    llama3 = dspy.HFModel(model = arguments.model)

    dspy.settings.configure(lm=llama3)
    print("LOADED LAMA")

    train_dataset = make_dataset(arguments.train_data,
                                arguments.train_labels)

    dev_dataset = make_dataset(arguments.dev_data,
                                arguments.dev_labels)
    print("LOADED DATASET")

    # dev_dataset = dev_dataset[:100]
    chain_of_thought = CoT()

    answers = {}

    for example in tqdm(dev_dataset):
        id_ = example.id_
        response = chain_of_thought(question=example.question, 
                                    choices=example.choices
                                    )
        answers[id_] = response._store

    with open(arguments.output, 'wb') as f:
        pickle.dump(answers, f)

    # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
    # config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

    # # Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
    # teleprompter = BootstrapFewShot(metric=accuracy_metric, **config)
    # optimized_cot = teleprompter.compile(CoT(), trainset=train_dataset, valset=dev_dataset)


    # evaluate = Evaluate(devset=dev_dataset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

    # Evaluate our `optimized_cot` program.
    # evaluate(optimized_cot)
    # llama3.inspect_history(n=1)
