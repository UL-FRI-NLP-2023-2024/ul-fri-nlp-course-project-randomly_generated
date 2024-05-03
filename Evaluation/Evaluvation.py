import json
from transformers import AutoTokenizer, AutoModel
import torch
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors

# Initialize the BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Load the Word2Vec model
w2v_model = KeyedVectors.load_word2vec_format("C:\\Users\\zan24\\Desktop\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin", binary=True)

# Load the JSONL files
with open('C:\\Users\\zan24\\Desktop\\questions_and_answers.jsonl', 'r') as f:
    questions_and_answers = [json.loads(line) for line in f]

# Load the input files into a list
inputs = []
for i in range(1, 3):  # Change the range according to the number of inputs
    with open(f'C:\\Users\\zan24\\Desktop\\input{i}.jsonl', 'r') as f:
        inputs.append([json.loads(line) for line in f])

# Initialize counters
correct_counts = [0] * len(inputs)
total_questions = 0

# Iterate over the questions and answers
for qa in questions_and_answers:
    similarity_weight = 0.8
    wmd_weight = 0.2

    total_questions += 1
    # Get the correct answer
    correct_answer = qa['correct_sol']

    # Find the corresponding answers in each input
    answers = [next((item for item in input if item["id"] == qa["id"]), None) for input in inputs]

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


    # Calculate the weighted sum of the cosine similarity and Word Mover's Distance
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

# Print the final counts
for i, count in enumerate(correct_counts):
    print(f'Final count of more similar answers for input{i+1}: {count}')
    print(f'Percentage of correct answers for input{i+1}: {count/total_questions*100}%')