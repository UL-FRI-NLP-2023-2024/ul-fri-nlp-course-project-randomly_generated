import json
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from gensim.models import KeyedVectors
import huggingface_hub

### This is an unstable experimantal script that works half the time 
### becouse of the unstability we switched to the stable version of the script
### it works about 50% if you take the time to set up the conflicting libraries

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
w2v_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
TOKEN = "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"
huggingface_hub.login(token=TOKEN)
gemm_tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b')
gemm_model = AutoModel.from_pretrained('google/gemma-7b')
model_name = 'google/gemma-7b'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('questions_and_answers.jsonl', 'r') as f:
    questions_and_answers = [json.loads(line) for line in f]

inputs = []
for i in range(1, 3):
    with open(f'input{i}.jsonl', 'r') as f:
        inputs.append([json.loads(line) for line in f])

correct_counts = [0] * len(inputs)
lm_counts = [0] * len(inputs)
total_questions = 0
lm_scores = []

for qa in questions_and_answers:
    similarity_weight = 0.4
    wmd_weight = 0.6
    total_questions += 1
    correct_answer = qa['correct_sol']
    answers = [next((item for item in input if item["id"] == qa["id"]), None) for input in inputs]
    if any(answer is None for answer in answers):
        continue
    correct_answer_tokens = tokenizer(correct_answer, return_tensors='pt', truncation=True, padding=True)
    answer_tokens = [tokenizer(answer['sol'], return_tensors='pt', truncation=True, padding=True) for answer in answers]
    with torch.no_grad():
        correct_answer_embedding = model(**correct_answer_tokens).last_hidden_state.mean(dim=1)
        answer_embeddings = [model(**tokens).last_hidden_state.mean(dim=1) for tokens in answer_tokens]
    similarities = [torch.nn.functional.cosine_similarity(correct_answer_embedding, embedding) for embedding in answer_embeddings]
    wmds = [w2v_model.wmdistance(correct_answer, answer['sol']) for answer in answers]
    scores = [similarity_weight * similarity - wmd_weight * wmd for similarity, wmd in zip(similarities, wmds)]
    for i, score in enumerate(scores):
        if score > 0.8 or score > 0.3:
            correct_counts[i] += 1
    for i in range(len(answers)):
        answer1 = answers[i]['sol']
        answer2 = answers[(i+1)%len(answers)]['sol'] if i+1 < len(answers) else None
        if answer2 is None:
            break
        prompt = f'Question: {qa["goal"]}\nCorrect Answer: {correct_answer}\n'
        for i, answer in enumerate(answers):
            prompt += f'Answer {i+1}: {answer["sol"]}\n'
        prompt += 'Which answer is more correct? Please respond with the answer number only. Do not provide any other answer.'
        input_ids = gemm_tokenizer.encode(prompt, return_tensors='pt')
        output_ids = gemm_model.generate(input_ids)
        model_answer = gemm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        lm_score = 0
        for i in range(len(answers)):
            if model_answer.lower() == f"answer {i+1}":
                lm_score = i+1
                break
        lm_counts[lm_score-1] += 1
        lm_scores.append(lm_score)

for i, count in enumerate(correct_counts):
    print(f'Final count of more similar answers for input{i+1}: {count}')
    print(f'Percentage of correct answers for input{i+1}: {count/total_questions*100}%')

for i, count in enumerate(lm_counts):
    print(f'Final count of lm scores for input{i+1}: {count}')
    print(f'Percentage of lm scores for input{i+1}: {count/total_questions*100}%')