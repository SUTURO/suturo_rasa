#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

import yaml

def parse_example(line: str):
    pattern = re.compile(r'\[(.*?)\]\{((?:[^{}]|{[^{}]*})*)\}')
    entities = []
    plain_parts = []
    last_end = 0

    for match in pattern.finditer(line):
        start, end = match.span()
        plain_parts.append(line[last_end:start])
        entity_text = match.group(1)
        attrs_str = match.group(2)

        # Parse the JSON-like attributes inside braces
        try:
            attrs = json.loads('{' + attrs_str + '}')
        except json.JSONDecodeError:
            # If malformed, skip this entity (should not happen in well-formed nlu.yml)
            attrs = {}

        # Build entity object: include value and all attributes from the annotation
        entity = {"value": entity_text}
        entity.update(attrs)   # adds 'entity', 'role', etc.
        entities.append(entity)

        plain_parts.append(entity_text)
        last_end = end

    plain_parts.append(line[last_end:])
    sentence = re.sub(r'\s+', ' ', ''.join(plain_parts)).strip()
    return sentence, entities

def convert_nlu_to_jsonl(input_file: Path, output_file: Path):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    out_lines = []
    for item in data.get('nlu', []):
        if 'lookup' in item:      # skip lookup tables
            continue
        if 'intent' not in item:
            continue

        intent_name = item['intent']
        examples = item.get('examples', '').strip().split('\n')

        for ex in examples:
            ex = ex.strip()
            if not ex or not ex.startswith('- '):
                continue
            ex = ex[2:].strip()   # remove leading dash

            sentence, entities = parse_example(ex)

            out_obj = {
                "sentence": sentence,
                "intent": intent_name,
                "entities": entities
            }
            out_lines.append(json.dumps(out_obj, ensure_ascii=False))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    print(f"Converted {len(out_lines)} examples to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_nlu_to_jsonl.py <nlu.yml> <output.jsonl>")
        sys.exit(1)
    convert_nlu_to_jsonl(Path(sys.argv[1]), Path(sys.argv[2]))