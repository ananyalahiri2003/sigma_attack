from pathlib import Path
import yaml
import json
import click


@click.command()
# @click.option("--input-dir",
#               type=str, default="/Users/ananya.lahiri/sigma/rules/windows", help="Path to data directory")
# @click.option("--output-dir",
#               type=str, default="/Users/ananya.lahiri/output_sigma/rules/windows", help="Path to output directory")
@click.option("--input-dir",
              type=str, default="/Users/ananya.lahiri/sigma/rules/windows/driver_load", help="Path to data directory")
@click.option("--output-dir",
              type=str, default="/Users/ananya.lahiri/output_sigma/temp/rules/windows/driver_load", help="Path to output directory")
def parse_yaml_files(
        input_dir: str,
        output_dir: str,
):
    print("Starting")
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    print(f"Created {output_dir=}")
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

        output_filename = f"{file_path.stem}.json"
        output_filepath = Path(output_dir)/output_filename
        with open(output_filepath, 'w') as filep:
            json.dump(parsed_data, filep, indent=4)
        print(f"Processed and saved")


if __name__ == "__main__":
    parse_yaml_files()


