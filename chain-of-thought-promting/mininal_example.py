import huggingface_hub

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot

from dspy.evaluate import Evaluate


TOKEN = "PASTE TOKEN FROM https://huggingface.co/settings/tokens"

huggingface_hub.login(token=TOKEN)

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer", temperature=0.01)
    
    def forward(self, question):
        return self.prog(question=question)

# print("login sucessful")

llama3 = dspy.HFModel(model = 'meta-llama/Meta-Llama-3-8B-Instruct')

dspy.settings.configure(lm=llama3)
print("LOADED LAMA")
print('--'*50)

gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]



# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)
print('-------------------')
print("EVAL END")
print("-----------------HISOTRY-----------")
llama3.inspect_history(n=1)