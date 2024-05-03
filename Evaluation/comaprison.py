import json

# Function to process a file
def process_file(input_file, output_file):
    # Open the JSON file and load the data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Open the output file
    with open(output_file, 'w') as f:
        # Iterate over the items in the data
        for id, item_data in data.items():
            # Get the "Number of the selection" field
            number_of_the_selection = item_data['Number of the selection']

            # Count the occurrences of 0 and 1
            count0 = number_of_the_selection.count('0')
            count1 = number_of_the_selection.count('1')

            # Check which count is higher and write the ID and the corresponding number to the file as a JSON object
            if count0 > count1:
                f.write(json.dumps({"id": id, "number": 0}) + '\n')
            elif count1 > count0:
                f.write(json.dumps({"id": id, "number": 1}) + '\n')

# Process the 'output2.json' file
process_file('C:\\Users\\zan24\\Desktop\\output2.json', 'output_COT.jsonl')

# Process the 'output_NO.json' file
process_file('C:\\Users\\zan24\\Desktop\\output_NO.json', 'output_NO.jsonl')

import json

# Function to calculate the score
def calculate_score(file_name):
    # Initialize the score and the count of answers
    score = 0
    count = 0

    # Open the file
    with open(file_name, 'r') as f1:
        # Read and process each line separately
        for line1 in f1:
            # Load the data from the line
            output_data = json.loads(line1)

            # Increment the count of answers
            count += 1

            # Open the 'output_id_number.jsonl' file
            with open("C:\\Users\\zan24\\Desktop\\output_id_number.jsonl", 'r') as f:
                # Read and process each line separately
                for line in f:
                    # Load the data from the line
                    output_id_number_data = json.loads(line)

                    # Check if the ID and number fields match, and increment the score if they do
                    if output_data["id"] == output_id_number_data["id"] and output_data["number"] == output_id_number_data["number"]:
                        score += 1
                        break  # No need to check the rest of the lines

    # Calculate the final score
    final_score = score / count if count > 0 else 0
    print(f'Score for {file_name}: {score}')
    print(f'Count for {file_name}: {count}')

    # Print the final score
    print(f'Final score for {file_name}: {final_score}')

# Calculate the score for the 'output_COT.jsonl' file
calculate_score('output_COT.jsonl')

# Calculate the score for the 'output_NO.jsonl' file
calculate_score('output_NO.jsonl')