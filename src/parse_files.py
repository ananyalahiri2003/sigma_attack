from pathlib import Path
import yaml
import json
import click
import datetime
import requests
from jsonschema import validate, ValidationError


@click.command()
# @click.option("--input-dir",
#               type=str, default="/Users/ananyalahiri/sigma_attack_windows/sigma/rules/windows/driver_load", help="Path to data directory")
# @click.option("--output-dir",
#               type=str, default="/Users/ananyalahiri/output_sigma/temp/rules/windows/driver_load", help="Path to output directory")
@click.option("--input-dir",
              type=str, default="/Users/ananyalahiri/sigma_attack_windows/sigma/rules/windows", help="Path to data directory")
@click.option("--output-dir",
              type=str, default="/Users/ananyalahiri/output_sigma/temp/rules/windows", help="Path to output directory")
@click.option("--schema-url",
              type=str,
              default="https://raw.githubusercontent.com/SigmaHQ/sigma/master/tests/validate-sigma-schema/sigma-schema.json",
              help="Path to JSON schema file")
@click.option("--reputation-threshold", type=int, default=20, help="Minimum author's reputation to include file")
def parse_yaml_files(
        input_dir: str,
        output_dir: str,
        schema_url: str,
        reputation_threshold: int = 0,
):
    print("Starting")
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    print(f"Created {output_dir=}")

    # Load the json schema
    try:
        response = requests.get(schema_url)
        schema = response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch schema: {e}")

    tag_dis = {}

    # Input directory loop through
    files = [fl for fl in Path(input_dir).rglob('*.yml')]
    print(f"files:\n{files}")

    for file_path in files:
        with open(file_path, 'r') as f:
            try:
                parsed_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error parsing file {file_path}: {e}")
                continue

            for fld in ['date', 'modified']:
                if fld in parsed_data and isinstance(parsed_data[fld], (datetime.date, datetime.datetime)):
                    parsed_data[fld] = parsed_data[fld].isoformat()

        # Validate against the schema
        try:
            validate(instance=parsed_data, schema=schema)
        except ValidationError as e:
            print(f"Schema validation error in file {file_path}: {e}")
            continue

        # Filter author reputation
        # author_reputation = parsed_data.get("author_reputation", 0)
        # if author_reputation < reputation_threshold:
        #     print(f"Author repute {author_reputation} < {reputation_threshold}, skipping")
        #     continue

        # Get tag stats
        tags = parsed_data.get('tags', [])
        for tag in tags:
            if tag in tag_dis:
                tag_dis[tag] += 1
            else:
                tag_dis[tag] = 1

        output_filename = f"{file_path.stem}.json"
        output_filepath = Path(output_dir)/output_filename
        with open(output_filepath, 'w') as filep:
            json.dump(parsed_data, filep, indent=4)
        print(f"Processed and saved")

    tags_dist_filepath = Path(output_dir) / "tag_distribution.json"
    with open(tags_dist_filepath, 'w') as f:
        json.dump(tag_dis, f, indent=4)
    print(f"Tag dists saved under {tags_dist_filepath}")

    print("Tag distn statistics...")
    for tag, count in tag_dis.items():
        print(f"{tag=}: {count=}")


if __name__ == "__main__":
    parse_yaml_files()


