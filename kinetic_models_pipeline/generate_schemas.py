import json
import os
from pathlib import Path

import requests
from datamodel_code_generator import InputFileType, generate


class MissingEnvironmentVariable(Exception):
    pass


def get_json_schema(url) -> str:
    content = requests.get(url).json()
    return json.dumps(content)


def generate_models(endpoint: str, path: Path = Path(__file__).parent / "models.py") -> None:
    json_schema = get_json_schema(endpoint)
    generate(
        json_schema,
        input_file_type=InputFileType.OpenAPI,
        input_filename="openapi.json",
        output=path,
    )


def gen():
    endpoint = os.getenv("SCHEMA_ENDPOINT")
    if endpoint is None:
        raise MissingEnvironmentVariable("SCHEMA_ENDPOINT is not set")
    generate_models(endpoint)
