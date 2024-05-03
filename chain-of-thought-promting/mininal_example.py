import json
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np
import huggingface_hub

import dspy

# from dspy.teleprompt import BootstrapFewShot
# from dspy.evaluate import Evaluate


TOKEN = "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"

huggingface_hub.login(token=TOKEN)



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

        choices = f"--0:{answ0}\n --1:{answ1}"
    
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


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought('question, choices -> reasoning, Number of the selection', 
                                        temperature=0.1)
    
    def forward(self, question, choices):
        return self.prog(question=question, choices=choices)


llama3 = dspy.HFModel(model = 'meta-llama/Meta-Llama-3-8B-Instruct')

dspy.settings.configure(lm=llama3)
print("LOADED LAMA")
print('--'*50)

train_dataset = make_dataset('/d/hpc/home/in7357/data/physicaliqa-train-dev/train.jsonl',
                             '/d/hpc/home/in7357/data/physicaliqa-train-dev/train-labels.lst')

dev_dataset = make_dataset('/d/hpc/home/in7357/data/physicaliqa-train-dev/dev.jsonl',
                             '/d/hpc/home/in7357/data/physicaliqa-train-dev/dev-labels.lst')
print("LOADED DATASET")
# train_dataset = train_dataset[:2]
dev_dataset = dev_dataset
print("TRAIN DATASET: ")
# print(train_dataset)
print('-----------------')
print("DEV DATASET:----------------")
# print(dev_dataset)
print('---'*10)
dev_dataset = dev_dataset[:100]
chain_of_thought = CoT()

answers = {}

for example in tqdm(dev_dataset):
    id_ = example.id_
    response = chain_of_thought(question=example.question, 
                                choices=example.choices
                                )
    answers[id_] = response._store
import pickle
with open('/d/hpc/home/in7357/ul-fri-nlp-course-project-randomly_generated/model_outputs/responses_COT.pick', 'wb') as f:
    pickle.dump(answers, f)

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
# config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# # Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
# teleprompter = BootstrapFewShot(metric=accuracy_metric, **config)
# optimized_cot = teleprompter.compile(CoT(), trainset=train_dataset, valset=dev_dataset)


# evaluate = Evaluate(devset=dev_dataset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
# evaluate(optimized_cot)
llama3.inspect_history(n=1)
