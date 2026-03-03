#!/usr/bin/env python3

import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ollama
from tqdm import tqdm

OUTPUT_PATH = Path("results")
ALLOWED_INTENTS = [
    "go",
    "guide",
    "take",
    "place",
    "deliver",
    "affirm",
    "deny",
    "open",
    "lookup",
    "talk",
    "seating",
    "receptionist",
]
ALLOWED_ENTITIES = [
    "NaturalPerson",
    "Room",
    "DesignedFurniture",
    "Clothing",
    "Transportable",
    "Interest",
    "food",
    "drink",
]
ALLOWED_ROLES = [
    "Person",
    "Location",
    "Furniture",
    "Clothes",
    "Item",
    "Hobby",
    "Food",
    "Drink",
]
REQUIRED_FIELDS = ["sentence", "intent", "entities"]


@dataclass
class GroundTruth:
    sentence: str
    intent: str
    entities: List[dict]


@dataclass
class ValidationCheck:
    valid_json: bool = False
    only_json: bool = False
    has_all_fields: bool = False
    intent_allowed: bool = False
    intent_correct: bool = False
    entities_allowed: bool = False
    entities_correct: bool = False
    roles_allowed: bool = False
    roles_correct: bool = False
    predicted_intent: Optional[str] = None
    predicted_entities: Optional[List[dict]] = None
    errors: List[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class SampleResult:
    sentence: str
    ground_truth: GroundTruth
    model: str
    response_time: float
    output: str
    validation: ValidationCheck


def load_ground_truth(file):
    gt = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            gt.append(GroundTruth(**data))
    print(f"Loaded {len(gt)} ground truth samples")
    return gt


# === Does it contain JSON? ===
def contain_json(text):
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    return match.group(1) if match else None


# === Is it correct JSON? ===
def parse_json(json_str):
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError:
        return None


# === Does it have valid intents? ===
def valid_intent(pred, gt):
    errors = []
    allowed = pred in ALLOWED_INTENTS
    correct = pred == gt

    if not allowed:
        errors.append(f"Intent '{pred}' not allowed.")
    if not correct:
        errors.append(f"Intent '{pred}' not correct. Expected: {gt}")
    return allowed, correct, errors


# === Does it have valid entities? ===
def valid_entities(pred, gt):
    pred_ent = sorted(str(e.get("entity") or "") for e in pred)
    pred_roles = sorted(str(e.get("role") or "") for e in pred)
    gt_ent = sorted(str(e.get("entity") or "") for e in gt)
    gt_roles = sorted(str(e.get("role") or "") for e in gt)

    bad_ent = [e for e in pred_ent if e not in ALLOWED_ENTITIES]
    bad_roles = [r for r in pred_roles if r not in ALLOWED_ROLES]

    errors = []
    if bad_ent:
        errors.append(f"Invalid entities: {bad_ent}.")
    if bad_roles:
        errors.append(f"Invalid roles: {bad_roles}.")
    if pred_ent != gt_ent:
        errors.append(f"Wrong entities: {pred_ent} (expected: {gt_ent}).")
    if pred_roles != gt_roles:
        errors.append(f"Wrong roles: {pred_roles} (expected: {gt_roles}).")

    return {
        "allowed_ent": not bad_ent,
        "allowed_roles": not bad_roles,
        "correct_ent": pred_ent == gt_ent,
        "correct_roles": pred_roles == gt_roles,
        "errors": errors,
    }


# === Gather everything into the ValidationCheck class ===
def validate_response(response, gt):
    check = ValidationCheck(raw_output=response)

    # === Containing JSON? ===
    json_str = contain_json(response)
    if not json_str:
        check.errors.append("No JSON object found.")
        return check
    check.only_json = response.strip() == json_str.strip()

    # === Valid JSON? ===
    parsed = parse_json(json_str)
    if not parsed:
        check.errors.append("Invalid JSON.")
        return check
    check.valid_json = True

    # === Are there missing fields in JSON? ===
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    if missing:
        check.errors.append(f"Missing required fields: {missing}")
    else:
        check.has_all_fields = True

    # === Does the JSON contain only allowed values? ===
    pred_intent = parsed.get("intent", "")
    check.predicted_intent = pred_intent
    pred_entities = parsed.get("entities", [])
    check.predicted_entities = pred_entities

    # === Valid intents ===
    int_allowed, int_correct, int_err = valid_intent(pred_intent, gt.intent)

    check.intent_allowed = int_allowed
    check.intent_correct = int_correct
    check.errors.extend(int_err)

    # === Valid entities ===
    valid_ent = valid_entities(pred_entities, gt.entities)
    check.entities_allowed = valid_ent["allowed_ent"]
    check.roles_allowed = valid_ent["allowed_roles"]
    check.entities_correct = valid_ent["correct_ent"]
    check.roles_correct = valid_ent["correct_roles"]
    check.errors.extend(valid_ent["errors"])

    return check


def stats(results):
    total = len(results)
    if total == 0:
        return {}

    def percentage(condition):
        return sum(1 for r in results if condition(r)) / total * 100

    intent_stats = {}
    for res in results:
        gt_intent = res.ground_truth.intent
        if gt_intent not in intent_stats:
            intent_stats[gt_intent] = {"total": 0, "correct": 0}
        intent_stats[gt_intent]["total"] += 1
        if res.validation.intent_correct:
            intent_stats[gt_intent]["correct"] += 1

    statistic = {
        "total": total,
        "valid_json": percentage(lambda r: r.validation.valid_json),
        "only_json": percentage(lambda r: r.validation.only_json),
        "has_all_fields": percentage(lambda r: r.validation.has_all_fields),
        "intent_allowed": percentage(lambda r: r.validation.intent_allowed),
        "intent_correct": percentage(lambda r: r.validation.intent_correct),
        "entities_allowed": percentage(lambda r: r.validation.entities_allowed),
        "entities_correct": percentage(lambda r: r.validation.entities_correct),
        "roles_allowed": percentage(lambda r: r.validation.roles_allowed),
        "roles_correct": percentage(lambda r: r.validation.roles_correct),
        "avg_time": sum(res.response_time for res in results) / total,
        "intent_acc": {
            intent: {
                "total": v["total"],
                "correct": v["correct"],
                "percent": v["correct"] / v["total"] * 100,
            }
            for intent, v in intent_stats.items()
        },
    }

    return statistic


def save(model, results, statistic):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUT_PATH.mkdir(exist_ok=True)
    path = OUTPUT_PATH / f"{model}_{ts}.json"
    err_path = OUTPUT_PATH / f"{model}_{ts}_errors.json"

    data = {
        "model": model,
        "timestamp": ts,
        "stats": statistic,
        "samples": [
            {
                "latency": res.response_time,
                "sentence": res.sentence,
                "ground_truth": {
                    "intent": res.ground_truth.intent,
                    "entities": res.ground_truth.entities,
                    # "roles": res.ground_truth.roles,
                },
                "predicted_intent": res.validation.predicted_intent,
                "predicted_entities": res.validation.predicted_entities,
                "validation": {
                    k: v
                    for k, v in asdict(res.validation).items()
                    if k not in ("predicted_intent", "predicted_entities", "raw_output")
                },
                "output": res.output,
            }
            for res in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    errors = [
        {
            "sentence": res.sentence,
            "ground_truth_intent": res.ground_truth.intent,
            "predicted_intent": res.validation.predicted_intent,
            "predicted_entities": res.validation.predicted_entities,
            "errors": res.validation.errors,
            "output": res.output,
        }
        for res in results
        if res.validation.errors
    ]
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model, "total_err": len(errors), "errors": errors},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved full results: {path}.")
    print(f"Saved errors: {err_path}.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gt", "--ground_truth", type=Path, help="Path to ground truth file."
    )
    parser.add_argument(
        "-m", "--models", nargs="+", required=True, help="LLM models to test"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default="./results/",
        help="Location to save the results.",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)

    for model in args.models:
        print(f"> Testing model: {model}")
        results = []
        try:
            for gt in tqdm(ground_truth, total=len(ground_truth), desc=model):
                start = time.perf_counter()
                try:
                    response = ollama.chat(
                        model=model,
                        messages=[{"role": "user", "content": gt.sentence}],
                        think=False,
                    )
                    raw = response.message.content.strip()
                except Exception as e:
                    raw = f"Error: {e}"
                    print(f"Error on '{gt.sentence[:40]}...': {e}")
                latency = time.perf_counter() - start

                validation = validate_response(raw, gt)
                results.append(
                    SampleResult(
                        sentence=gt.sentence,
                        ground_truth=gt,
                        model=model,
                        response_time=latency,
                        output=raw,
                        validation=validation,
                    )
                )
        except KeyboardInterrupt:
            print(f"\nTry saving results in {args.output}.")
            if results:
                statistics = stats(results)
                save(model, results, statistics)
            else:
                print("No results to save.")
            sys.exit(0)

        statistics = stats(results)
        save(model, results, statistics)


if __name__ == "__main__":
    main()
