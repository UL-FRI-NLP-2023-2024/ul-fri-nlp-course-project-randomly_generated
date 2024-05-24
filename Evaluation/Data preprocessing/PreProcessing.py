import json
import re
import os
from collections import OrderedDict

class JsonlProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def convert_str_to_int(self, field):
        for file_path in self.file_paths:
            data = []
            with open(file_path, 'r', encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if field in obj and isinstance(obj[field], str):
                        try:
                            obj[field] = int(obj[field])
                        except ValueError:
                            pass
                    data.append(obj)
            with open(file_path, 'w') as f:
                for obj in data:
                    f.write(json.dumps(obj) + '\n')

    def rename_field(self, old_field, new_field):
        for file_path in self.file_paths:
            data = []
            with open(file_path, 'r', encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if old_field in obj:
                        obj[new_field] = obj.pop(old_field)
                    data.append(obj)
            with open(file_path, 'w') as f:
                for obj in data:
                    f.write(json.dumps(obj) + '\n')

    def add_field_from_string(self, data_field, pattern, new_field):
        for file_path in self.file_paths:
            data = []
            with open(file_path, 'r', encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    match = re.search(pattern, obj.get(data_field, ''))
                    if match:
                        obj[new_field] = match.group(1)
                    data.append(obj)
            with open(file_path, 'w') as f:
                for obj in data:
                    f.write(json.dumps(obj) + '\n')

    def add_index_field(self):
        for file_path in self.file_paths:
            data = []
            with open(file_path, 'r', encoding="utf-8") as f:
                for index, line in enumerate(f):
                    obj = json.loads(line)
                    obj['index'] = index
                    data.append(obj)
            with open(file_path, 'w') as f:
                for obj in data:
                    f.write(json.dumps(obj) + '\n')

###Usage example ####
processor = JsonlProcessor(['file1.jsonl', 'file2.jsonl'])
processor.convert_str_to_int('answer')
processor.rename_field('old_field', 'new_field')
processor.add_field_from_string('raw', r'\nAnswer: (\d)', 'answer')
processor.add_index_field()