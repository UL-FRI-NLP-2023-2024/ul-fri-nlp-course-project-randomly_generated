import dspy
import huggingface_hub
from tqdm import tqdm 

TOKEN = "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"

huggingface_hub.login(token=TOKEN)

#mistralai/Mistral-7B-Instruct-v0.2
# This sets up the language model for DSPy in this case we are using mistral 7b through TGI (Text Generation Interface from HuggingFace)
llm = dspy.HFModel(model='google/gemma-7b')

#mistral.kwargs["temperature"] = 0.5
# This sets the language model for DSPy.
dspy.settings.configure(lm=llm)

# This is not required but it helps to understand what is happening
my_example = {
    "question": "What system was final fantasy 1 made for?",
    "answer": "It was made for NES",
}

# This is the signature for the predictor. It is a simple question and answer model.
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 6 and 10 words")



for i in tqdm(range(10)):

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    pred = generate_answer(question=my_example['question'])


generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
pred = generate_answer(question=my_example['question'])

# Print the answer...profit :)
print(pred.answer)
