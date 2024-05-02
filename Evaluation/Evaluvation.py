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
with open('C:\\Users\\zan24\\Desktop\\input1.jsonl', 'r') as f:
    input1 = [json.loads(line) for line in f]
with open('C:\\Users\\zan24\\Desktop\\input2.jsonl', 'r') as f:
    input2 = [json.loads(line) for line in f]

# Initialize counters
correct_count1 = 0
correct_count2 = 0

# Iterate over the questions and answers
for qa in questions_and_answers:
    # Get the correct answer
    correct_answer = qa['correct_sol']

    # Find the corresponding answers in input1 and input2
    answer1 = next((item for item in input1 if item["id"] == qa["id"]), None)
    answer2 = next((item for item in input2 if item["id"] == qa["id"]), None)

    # Skip if either answer is not found
    if not answer1 or not answer2:
        continue

    # Tokenize and encode the correct answer and the two new answers
    correct_answer_tokens = tokenizer(correct_answer, return_tensors='pt', truncation=True, padding=True)
    answer1_tokens = tokenizer(answer1['sol'], return_tensors='pt', truncation=True, padding=True)
    answer2_tokens = tokenizer(answer2['sol'], return_tensors='pt', truncation=True, padding=True)

    # Get the BERT embeddings for the correct answer and the two new answers
    with torch.no_grad():
        correct_answer_embedding = model(**correct_answer_tokens).last_hidden_state.mean(dim=1)
        answer1_embedding = model(**answer1_tokens).last_hidden_state.mean(dim=1)
        answer2_embedding = model(**answer2_tokens).last_hidden_state.mean(dim=1)

    # Calculate the cosine similarity of the new answers with the correct answer
    similarity1 = torch.nn.functional.cosine_similarity(correct_answer_embedding, answer1_embedding)
    similarity2 = torch.nn.functional.cosine_similarity(correct_answer_embedding, answer2_embedding)

    # Calculate the Word Mover's Distance of the new answers with the correct answer
    wmd1 = w2v_model.wmdistance(correct_answer, answer1['sol'])
    wmd2 = w2v_model.wmdistance(correct_answer, answer2['sol'])

    # Define weights for the cosine similarity and Word Mover's Distance
    similarity_weight = 0.3
    wmd_weight = 0.7

    # Calculate the weighted sum of the cosine similarity and Word Mover's Distance
    score1 = similarity_weight * similarity1 - wmd_weight * wmd1
    score2 = similarity_weight * similarity2 - wmd_weight * wmd2

    # Choose the answer with the higher score as the correct one
    if score1 > score2:
        print(f'For the question: "{qa["goal"]}"\nCorrect answer: "{correct_answer}"\nChosen answer from input1: "{answer1["sol"]}"')
        correct_count1 += 1
    else:
        print(f'For the question: "{qa["goal"]}"\nCorrect answer: "{correct_answer}"\nChosen answer from input2: "{answer2["sol"]}"')
        correct_count2 += 1

# Print the final counts
print(f'Final count of more similar answers: input1={correct_count1}, input2={correct_count2}')