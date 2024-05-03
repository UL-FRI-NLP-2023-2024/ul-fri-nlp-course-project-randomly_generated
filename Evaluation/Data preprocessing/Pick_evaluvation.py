import pickle
import json

# Define the file path
file_path = "C:\\Users\\zan24\\Desktop\\ul-fri-nlp-course-project-randomly_generated\\model_outputs\\responses_baseline_NO_REASONING.pick"

# Open and load the pickle file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Convert the data to JSON
json_data = json.dumps(data, indent=4)

# Define the output file path
output_file_path = "C:\\Users\\zan24\\Desktop\\output_NO.json"

# Open the output file and write the JSON data
with open(output_file_path, 'w') as f:
    f.write(json_data)