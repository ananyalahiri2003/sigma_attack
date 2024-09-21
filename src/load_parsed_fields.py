import json
import click
import re
import pandas as pd
from pathlib import Path
from typing import List


def load_parsed_data(input_dir):
    parsed_list = []
    files = [fl for fl in Path(input_dir).glob("*.json")]
    for fl in files:
        with open(fl, 'r') as f:
            data = json.load(f)
            parsed_list.append(data)
    return parsed_list


def truncate_tag(tag):
    if '.' in tag:
        parts = tag.split('.')
        if len(parts) > 2:
            return '.'.join(parts[:2])  # Keep only the first two components
    return tag


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
        try:
            combined = datum["description"]
            if 'falsepositives' in datum:
                combined = f"{combined} {datum['falsepositives']}"

            combined = clean_text(combined)
            record['text'] = combined
        except Exception as e:
            print(f"Could not created combined text: {e}")
            print(f"Without description {datum=}")
            continue

        for field in fields:
            if field == "tags" and field in datum:
                record[field] = [truncate_tag(tag) for tag in datum[field]]
            elif field in datum:
                record[field] = datum[field]
            else:
                record[field] = None

        extracted.append(record)

    return extracted


def save_extracted_data(extracted_data, output_dir, output_file_name="extracted_data.json"):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_file = Path(output_dir) / output_file_name
    try:
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=4)
            print(f"Extracted data saved to {output_file}")
    except Exception as e:
        raise ValueError(f"Could not save file: {e}")


def get_dataframe(extracted_data):
    df = pd.DataFrame(extracted_data)
    df_logsource = pd.json_normalize(df['logsource'])
    df_detection= pd.json_normalize(df['detection'])
    df = pd.concat([df.drop(['logsource', 'detection'], axis=1), df_logsource, df_detection], axis=1)
    return df


@click.command()
@click.option("--input-dir",
              type=str,
              default="/Users/ananyalahiri/output_sigma/temp/rules/windows",
              help="Input data directory"
              )
@click.option("--fields-to-extract",
              type=str,
              required=True,
              multiple=True,
              help="Fields to extract - I extract title, description, falsepositives, logsource, detection, tags"
              )
@click.option("--output-dir",
              type=str,
              default="Users/ananyalahiri/output_sigma/selected_fields_driver_load/rules/windows",
              help="Output directory")
@click.option("--output-file-name",
              type=str,
              default="extracted_data.csv",
              help="Name of output csv file")
def extract_data(input_dir,
                 fields_to_extract,
                 output_dir,
                 output_file_name):

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    try:
        parsed_data_list = load_parsed_data(input_dir)
    except Exception as e:
        raise ValueError("Check input directory, could not load")

    fields_to_extract = list(fields_to_extract)
    print(f"{fields_to_extract=}")
    extracted_data = extract_data_points(parsed_data_list, fields_to_extract) # At this stage we are still dealing with json data

    df = get_dataframe(extracted_data)

    # save_extracted_data(df, output_dir)
    output_filename = Path(output_dir)/output_file_name
    df.to_csv(output_filename, index=False)
    print(f"saved file under {output_filename}")


if __name__ == "__main__":
    extract_data()




