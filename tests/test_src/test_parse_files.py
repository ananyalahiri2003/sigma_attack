import json
from pathlib import Path
import pytest
import requests
from jsonschema import validate, ValidationError


SCHEMA_ROOT = Path("tests/test_data/schema.json")
SCHEMA_URL = "https://raw.githubusercontent.com/SigmaHQ/sigma/master/tests/validate-sigma-schema/sigma-schema.json"


def get_desired_schema():
    with open(SCHEMA_ROOT, 'r') as f:
        schema = json.load(f)
        return schema


def test_get_schema():

    desired = get_desired_schema()

    response = requests.get(SCHEMA_URL)
    output = response.json()

    assert desired == output



