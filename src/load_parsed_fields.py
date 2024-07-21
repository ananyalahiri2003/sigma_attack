import json
import click
import re
from pathlib import Path


def load_parsed_data(input_dir):
    parsed_list = []
    files = [fl for fl in Path(input_dir).glob("*.json")]
    for fl in files:
        with open(fl, 'r') as f:
            data = json.load(f)
            parsed_list.append(data)
    return parsed_list


# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def extract_data_points(parsed_list, fields):
    extracted = []
    for datum in parsed_list:
        record = {}
        for field in fields:
            if field in datum:
                record[field] = datum[field]
            else:
                record[field] = None
        try:
            description = datum["description"]
            falsepositives = datum["falsepositives"]
            combined = f"{description} {falsepositives}"
            combined = clean_text(combined)
            record['text'] = combined
        except Exception as e:
            print(f"Could not created combined text: {e}")
        extracted.append(record)
    return extracted


def save_extracted_data(extracted_data, output_dir, output_file_name="extracted_data.json"):
    output_dir = Path(output_dir).resolve()
    print(f"Resolved output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / output_file_name
    try:
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=4)
            print(f"Extracted data saved to {output_file}")
    except Exception as e:
        raise ValueError(f"Could not save file: {e}")


@click.command()
@click.option("--input-dir",
              type=str,
              default="/Users/ananya.lahiri/output_sigma/rules/windows/driver_load",
              help="Input data directory"
              )
@click.option("--fields-to-extract",
              type=str,
              required=True,
              multiple=True,
              help="Fields to extract"
              )
@click.option("--output-dir",
              type=str,
              default="Users/ananya.lahiri/output_sigma/selected_fields_driver_load/rules/windows/driver_load",
              help="Output directory")
def extract_data(input_dir,
                 fields_to_extract,
                 output_dir):

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        parsed_data_list = load_parsed_data(input_dir)
    except Exception as e:
        raise ValueError("Check input directory, could not load")

    fields_to_extract = list(fields_to_extract)
    print(f"{fields_to_extract=}")
    extracted_data = extract_data_points(parsed_data_list, fields_to_extract)

    save_extracted_data(extracted_data, output_dir)


if __name__ == "__main__":
    extract_data()




