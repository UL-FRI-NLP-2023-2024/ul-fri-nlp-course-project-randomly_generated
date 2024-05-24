This script is used to evaluate the performance of different models. It reads JSONL files containing model outputs, calculates scores for each model, and generates bar plots of the scores and the number of unanswered questions.

The script defines several global variables:

- INPUT_FILES: A list of paths to the JSONL files to be processed.
- MASTER_FILE: The path to the master file against which the model outputs are compared.
- FIELD_TO_COMPARE, ANSWER_FIELD, TARGET_FIELD: The names of the fields to be compared in the JSON objects.
- SCORES_PLOT_FILE, UNANSWERED_PLOT_FILE: The names of the files where the bar plots are saved.
- SCORES_PLOT_TITLE, UNANSWERED_PLOT_TITLE: The titles of the bar plots.

USAGE
Simply run the script. The script will process the files specified in INPUT_FILES, calculate the scores, generate the plots, and print the name and score of the best model.