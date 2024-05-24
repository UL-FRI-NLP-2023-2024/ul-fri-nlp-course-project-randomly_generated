import json
import os
import matplotlib.pyplot as plt

# Define global variables
INPUT_FILES = [
    r'C:\Users\zan24\Desktop\ul-fri-nlp-course-project-randomly_generated\model_outputs\anli\Baseline.jsonl',
    r'C:\Users\zan24\Desktop\ul-fri-nlp-course-project-randomly_generated\model_outputs\anli\CoT.jsonl',
    r'C:\Users\zan24\Desktop\ul-fri-nlp-course-project-randomly_generated\model_outputs\anli\CoT-H.jsonl',
    r'C:\Users\zan24\Desktop\ul-fri-nlp-course-project-randomly_generated\model_outputs\anli\FewShot.jsonl'
]
MASTER_FILE = "C:\\Users\\zan24\\Desktop\\ul-fri-nlp-course-project-randomly_generated\\transformed_data_anli_validation.jsonl"
FIELD_TO_COMPARE = "index"
ANSWER_FIELD = "answer"
TARGET_FIELD = "targets"
SCORES_PLOT_FILE = 'scores_plot_anli.png'
UNANSWERED_PLOT_FILE = 'unanswered_plot_anli.png'
SCORES_PLOT_TITLE = 'Scores of 300 questions'
UNANSWERED_PLOT_TITLE = 'Number of Unanswered Questions for Each Input File'

def process_file(input_file):
    unanswered = 0
    selections = []

    with open(input_file, 'r', encoding= 'utf-8') as f:
        for line in f:
            item_data = json.loads(line)
            selection = item_data.get(ANSWER_FIELD, 'N/A (needs further processing)')

            if selection == 'N/A (needs further processing)' or selection is None:
                unanswered += 1
            else:
                selections.append({FIELD_TO_COMPARE: item_data[FIELD_TO_COMPARE], ANSWER_FIELD: selection})

    return selections, unanswered

def calculate_score(file_path, selections, unanswered):
    score = 0
    count = 0
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(MASTER_FILE, 'r') as f:
        for line in f:
            output_id_number_data = json.loads(line)

            for selection in selections:
                if selection[FIELD_TO_COMPARE] == output_id_number_data[FIELD_TO_COMPARE]:
                    count += 1
                    if selection[ANSWER_FIELD] == output_id_number_data[TARGET_FIELD]:
                        score += 1
                    break

    final_score = score / count if count > 0 else 0
    print(f'Score for {file_name}: {score}')
    print(f'Count for {file_name}: {count}')
    print(f'Unanswered for {file_name}: {unanswered}')
    print(f'Final score for {file_name}: {final_score}')

    return file_name, final_score

scores = []
names = []
unanswered_counts = []

for input_file in INPUT_FILES:
    selections, unanswered = process_file(input_file)
    name, score = calculate_score(input_file, selections, unanswered)
    scores.append(score)
    names.append(name)
    unanswered_counts.append(unanswered) 

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  
bars = plt.bar(range(len(names)), scores, color=colors[:len(names)])  
plt.xlabel('Input Files')
plt.ylabel('Scores')
plt.title(SCORES_PLOT_TITLE)
plt.xticks([])  

plt.legend(bars, names, loc='upper left')
plt.savefig(SCORES_PLOT_FILE, bbox_inches='tight')
plt.show()

bars = plt.bar(range(len(names)), unanswered_counts, color=colors[:len(names)])  
plt.xlabel('Input Files')
plt.ylabel('Unanswered Questions')
plt.title(UNANSWERED_PLOT_TITLE)
plt.xticks([])  

plt.legend(bars, names, loc='upper right')
plt.savefig(UNANSWERED_PLOT_FILE, bbox_inches='tight')
plt.show()

best_index = scores.index(max(scores))
print(f'The best input file is {names[best_index]} with a score of {scores[best_index]}')