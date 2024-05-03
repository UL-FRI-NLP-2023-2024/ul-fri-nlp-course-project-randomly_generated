import json

# Open the JSONL file, the list file, and the output file
with open('dev.jsonl', 'r') as json_file, open('dev-labels.lst', 'r') as list_file, open('output_id_number.jsonl', 'w') as output_file:
    # Iterate over the lines in both files together
    for json_line, list_line in zip(json_file, list_file):
        # Parse the JSON line and get the list value
        data = json.loads(json_line)
        list_value = int(list_line.strip())

        # Write the id and the list value to the output file
        output_data = {'id': data['id'], 'number': list_value}
        output_file.write(json.dumps(output_data) + '\n')