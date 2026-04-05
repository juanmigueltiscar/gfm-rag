import json
import os
import re

KG_DELIMITER = ","

#itc: he comentat la funció que no guarda accents
def processing_phrases(phrase: str) -> str:
    if isinstance(phrase, int):
        return str(phrase)  # deal with the int values

    # Expresión regular que incluye letras acentuadas y otros caracteres especiales
    pattern = r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ ]"
    return re.sub(pattern, " ", phrase.lower()).strip()
'''
def processing_phrases(phrase: str) -> str:
    if isinstance(phrase, int):
        return str(phrase)  # deal with the int values
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()
'''

def directory_exists(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_json_dict(text: str) -> str | dict:
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ""
    else:
        return ""
