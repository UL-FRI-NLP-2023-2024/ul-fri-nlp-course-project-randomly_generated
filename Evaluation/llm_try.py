import json
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
import dspy
import huggingface_hub
from tqdm import tqdm 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Load the Word2Vec model
w2v_model = KeyedVectors.load_word2vec_format("C:\\Users\\zan24\\Desktop\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin", binary=True)

# Login to Hugging Face with the provided token
TOKEN = "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"
huggingface_hub.login(token=TOKEN)

# Initialize GEMM model and tokenizer
gemm_tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b')
gemm_model = AutoModel.from_pretrained('google/gemma-7b')

# Set up the language model
model_name = 'google/gemma-7b'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the JSONL files
with open('C:\\Users\\zan24\\Desktop\\questions_and_answers.jsonl', 'r') as f:
    questions_and_answers = [json.loads(line) for line in f]
    #print("Questions and Answers:", questions_and_answers)

# Load the input files into a list
inputs = []
for i in range(1, 3):  # Change the range according to the number of inputs
    with open(f'C:\\Users\\zan24\\Desktop\\input{i}.jsonl', 'r') as f:
        inputs.append([json.loads(line) for line in f])
#print("Inputs:", inputs)

# Initialize counters
correct_counts = [0] * len(inputs)
lm_counts = [0] * len(inputs)  # Initialize a counter for the lm scores
total_questions = 0
lm_scores = []

# Iterate over the questions and answers
for qa in questions_and_answers:
    similarity_weight = 0.8
    wmd_weight = 0.2
    total_questions += 1

    # Get the correct answer
    correct_answer = qa['correct_sol']
    #print("Processing question:", qa['goal'])

    # Find the corresponding answers in each input
    answers = [next((item for item in input if item["id"] == qa["id"]), None) for input in inputs]
    #print("Answers:", answers)
    # Skip if any answer is not found
    if any(answer is None for answer in answers):
        continue

    # Tokenize and encode the correct answer and the new answers
    correct_answer_tokens = tokenizer(correct_answer, return_tensors='pt', truncation=True, padding=True)
    answer_tokens = [tokenizer(answer['sol'], return_tensors='pt', truncation=True, padding=True) for answer in answers]

    # Get the BERT embeddings for the correct answer and the new answers
    with torch.no_grad():
        correct_answer_embedding = model(**correct_answer_tokens).last_hidden_state.mean(dim=1)
        answer_embeddings = [model(**tokens).last_hidden_state.mean(dim=1) for tokens in answer_tokens]

    # Calculate the cosine similarity of the new answers with the correct answer
    similarities = [torch.nn.functional.cosine_similarity(correct_answer_embedding, embedding) for embedding in answer_embeddings]

    # Calculate the Word Mover's Distance of the new answers with the correct answer
    wmds = [w2v_model.wmdistance(correct_answer, answer['sol']) for answer in answers]

    # Calculate the weighted sum of the cosine similarity, Word Mover's Distance, and language model scores
    scores = [similarity_weight * similarity - wmd_weight * wmd for similarity, wmd in zip(similarities, wmds)]
    # Choose the answer with the higher score as the correct one
    for i, score in enumerate(scores):
        print(f'Score for input{i+1}: {score}')
        if score > 0.8:
            print(f'For the question: "{qa["goal"]}"\nCorrect answer: "{correct_answer}"\nChosen answer from input{i+1}: "{answers[i]["sol"]}"')
            correct_counts[i] += 1
        elif score > 0.3:
            print(f'For the question: "{qa["goal"]}"\nCorrect answer: "{correct_answer}"\nChosen answer from input{i+1}: "{answers[i]["sol"]}"')
            correct_counts[i] += 1
        else:
            print(f'For the question: "{qa["goal"]}"\nCorrect answer: "{correct_answer}"\nNo correct answer found in input{i+1}.')


    for i in range(len(answers)):
        # Prepare two possible answers
        answer1 = answers[i]['sol']
        answer2 = answers[(i+1)%len(answers)]['sol'] if i+1 < len(answers) else None
        
        if answer2 is None:
            print("Not enough answers to compare.")
            break

        # Prepare the prompt
        prompt = f'Question: {qa["goal"]}\nCorrect Answer: {correct_answer}\nAnswer 1: {answer1}\nAnswer 2: {answer2}\nWhich answer is more correct, "Answer 1" or "Answer 2"? Please respond with "Answer 1" or "Answer 2" only. Do not provide any other answer.'

            # Tokenize the inputs using the GEMM tokenizer
        input_ids = gemm_tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response using the GEMM model
        output_ids = gemm_model.generate(input_ids)

        # Decode the output tokens into text using the GEMM tokenizer
        model_answer = gemm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'Model answer for input{i+1}: {model_answer}')

        # Score the model's answer: 1 if it matches "Answer 1", 2 if it matches "Answer 2", 0 if it matches neither
        if model_answer.lower() == "answer 1":
            lm_score = 1
        elif model_answer.lower() == "answer 2":
            lm_score = 2
        else:
            lm_score = 0

        # Increment the lm count for the current input
        lm_counts[i] += lm_score

        # Append the score to the list
        lm_scores.append(lm_score)

# Print the final counts
for i, count in enumerate(correct_counts):
    print(f'Final count of more similar answers for input{i+1}: {count}')
    print(f'Percentage of correct answers for input{i+1}: {count/total_questions*100}%')

# Print the final lm counts
for i, count in enumerate(lm_counts):
    print(f'Final count of lm scores for input{i+1}: {count}')
    print(f'Percentage of lm scores for input{i+1}: {count/total_questions*100}%')