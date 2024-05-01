import huggingface_hub

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot

from dspy.evaluate import Evaluate


TOKEN = "Hugginface account token"

huggingface_hub.login(token=TOKEN)

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

# print("login sucessful")

llama3 = dspy.OllamaLocal('llama3:8b')
dspy.settings.configure(lm=llama3)

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


# # Set up the LM
# turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
# dspy.settings.configure(lm=turbo)

# # Load math questions from the GSM8K dataset
# gsm8k = GSM8K()
# gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

