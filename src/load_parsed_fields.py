import json
from pathlib import Path


def load_parsed_data(input_dir):
    parsed_list = []
    files = [fl for fl in Path(input_dir).glob("*.json")]
    for fl in files:
        with open(fl, 'r') as f:
            data = json.load(f)
            parsed_list.append(data)
    return parsed_list


def extract_data_points(parsed_list, fields):
    extracted = []
    for datum in parsed_list:
        record = {}
        for field in fields:
            if field in datum:
                record[field] = datum[field]
            else:
                record[field] = None
        extracted.append(record)
    return extracted


input_dir = "/Users/ananya.lahiri/output_sigma/rules/windows/driver_load"
parsed_data_list = load_parsed_data(input_dir)

fields_to_extract = ["title", "description", "author", "tags", "logsource"]

extracted_data = extract_data_points(parsed_data_list, fields_to_extract)


for record in extracted_data:
    print(record)




