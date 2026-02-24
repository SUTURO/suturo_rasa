"""
Use nlu data from rasa and use the sentences as llm prompts and check if the results are good.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import ollama
import yaml
from tqdm import tqdm

OUTPUT_PATH = Path("results")
NLU = (Path(__file__).parent / "../MS3/data/nlu.yml").resolve()
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
]
ALLOWED_ENTITIES = [
    "NaturalPerson",
    "Room",
    "DesignedFurniture",
    "Clothing",
    "Transportable",
]
ALLOWED_ROLES = ["Person", "Location", "Furniture", "Clothes", "Item"]
REQUIRED_FIELDS = ["instruction", "sentence", "response", "intent", "entities"]

MODELS = [
    "nlp_llama",
    "nlp_llama_few",
    "nlp_llama_cot",
    "nlp_qwen",
    "nlp_qwen_few",
    "nlp_qwen_cot",
]


def load_ground_truth(nlu_file):
    with open(nlu_file, "r") as f:
        data = yaml.safe_load(f)

    samples = []
    for item in data.get("nlu", []):
        if "intent" not in item:
            continue

        intent = item["intent"]

        for line in item.get("examples", "").strip().split("\n"):
            line = line.strip()
            if not line.startswith("- "):
                continue

            raw = line[2:]

            entities = []
            for match in re.finditer(r"\[([^\]]+)\]\{([^}]+)\}", raw):
                value = match.group(1)
                meta = json.loads("{" + match.group(2) + "}")
                entities.append(
                    {
                        "value": value,
                        "entity": meta.get("entity", ""),
                        "role": meta.get("role", ""),
                    }
                )

            sentence = re.sub(r"\[([^\]]+)\]\{[^}]+\}", r"\1", raw).strip()

            samples.append(
                {
                    "sentence": sentence,
                    "intent": intent,
                    "entities": entities,
                }
            )

    return samples


def validate(reply, gt):
    """
    Check if llm response has the correct form and content
    """
    result = {
        "valid_json": False,
        "json_only": False,
        "all_fields": False,
        "allowed_intent": False,
        "correct_intent": False,
        "allowed_entity": False,
        "correct_entity": False,
        "allowed_roles": False,
        "correct_roles": False,
        "predicted_intent": None,
        "errors": [],
    }

    # Only JSON?
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        result["errors"].append(f"No JSON found.")
        return result

    json_str = match.group()

    if reply.strip() == json_str.strip():
        result["json_only"] = True
    else:
        result["errors"].append(f"Not ONLY JSON found.")

    # Valid JSON?
    try:
        parsed = json.loads(reply)
    except (json.decoder.JSONDecodeError, Exception) as e:
        result["errors"].append(f"Not valid JSON: {e}.")
        return result
    result["valid_json"] = True

    # All fields?
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    if missing:
        result["errors"].append(f"Missing field: {missing}")
    else:
        result["all_fields"] = True

    # Allowed intent and correct?
    pred_intent = parsed.get("intent", "")
    result["predicted_intent"] = pred_intent

    result["allowed_intent"] = pred_intent in ALLOWED_INTENTS
    result["correct_intent"] = pred_intent == gt["intent"]

    if not result["allowed_intent"]:
        result["errors"].append(f"Invalid intent: '{pred_intent}'")
    if not result["correct_intent"]:
        result["errors"].append(
            f"Wrong intent: '{pred_intent}' (expected: '{gt['intent']}')"
        )

    # 4. Allowed entities and roles and correct?
    pred_entities = parsed.get("entities", [])
    gt_entities = gt.get("entities", [])

    pred_entities_ = sorted(e.get("entity", "") for e in pred_entities)
    pred_roles = sorted(e.get("role", "") for e in pred_entities)
    gt_types = sorted(e.get("entity", "") for e in gt_entities)
    gt_roles = sorted(e.get("role", "") for e in gt_entities)

    bad_types = [t for t in pred_entities_ if t not in ALLOWED_ENTITIES]
    bad_roles = [r for r in pred_roles if r not in ALLOWED_ROLES]

    result["allowed_entity"] = not bad_types
    result["allowed_roles"] = not bad_roles
    result["correct_entity"] = pred_entities_ == gt_types
    result["correct_roles"] = pred_roles == gt_roles

    if bad_types:
        result["errors"].append(f"Invalid entity types: {bad_types}")
    if bad_roles:
        result["errors"].append(f"Invalid roles: {bad_roles}")
    if not result["correct_entity"]:
        result["errors"].append(
            f"Wrong entities: {pred_entities_} (expected: {gt_types})"
        )
    if not result["correct_roles"]:
        result["errors"].append(f"Wrong roles: {pred_roles} (expected: {gt_roles})")

    return result


def save_results(model, results, ts):
    OUTPUT_PATH.mkdir(exist_ok=True)
    path = OUTPUT_PATH / f"{model}_{ts}"

    total = len(results)

    keys = [
        "valid_json",
        "json_only",
        "all_fields",
        "allowed_intent",
        "correct_intent",
        "allowed_entity",
        "correct_entity",
        "allowed_roles",
        "correct_roles",
    ]

    summary = {
        key: {
            "count": sum(1 for r in results if r["validation"][key]),
            "percent": round(
                sum(1 for r in results if r["validation"][key]) / total * 100, 1
            ),
        }
        for key in keys
    }

    intent_accuracy = {}
    for r in results:
        gt = r["ground_truth"]["intent"]
        intent_accuracy.setdefault(gt, {"correct": 0, "total": 0})
        intent_accuracy[gt]["total"] += 1
        if r["validation"]["correct_intent"]:
            intent_accuracy[gt]["correct"] += 1

    for s in intent_accuracy.values():
        s["percent"] = round(s["correct"] / s["total"] * 100, 1)

    main_output = {
        "model": model,
        "timestamp": ts,
        "total_sentences": total,
        "summary": summary,
        "intent_accuracy": intent_accuracy,
        "results": [
            {
                "sentence": r["sentence"],
                "ground_truth": r["ground_truth"]["intent"],
                "predicted": r["validation"]["predicted_intent"],
                "correct": r["validation"]["correct_intent"],
                "validation": r["validation"],
                "raw_llm_output": r["raw_output"],
            }
            for r in results
        ],
    }

    main_path = Path(str(path) + ".json")
    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main_output, f, indent=2, ensure_ascii=False)

    errors = [
        {
            "sentence": r["sentence"],
            "ground_truth": r["ground_truth"]["intent"],
            "predicted": r["validation"]["predicted_intent"],
            "errors": r["validation"]["errors"],
            "raw_llm_output": r["raw_output"],
        }
        for r in results
        if r["validation"]["errors"]
    ]

    error_path = Path(str(path) + "_errors.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model, "total_errors": len(errors), "errors": errors},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"  Result: {main_path}")
    print(f"  Errors: {error_path}  ({len(errors)} from {total} sentences)\n")


def print_stats(model: str, results: list[dict]):
    total = len(results)

    def pct(key):
        n = sum(1 for r in results if r["validation"][key])
        return f"{n}/{total} ({n / total * 100:.1f}%)"

    print(f"\n{'=' * 50}")
    print(f"Model: {model}")
    print(f"{'=' * 50}")
    print(f"Valid JSON: {pct('valid_json')}")
    print(f"All fields: {pct('all_fields')}")
    print(f"Allowed intent: {pct('allowed_intent')}")
    print(f"Correct intent: {pct('correct_intent')}")
    print(f"Allowed entities: {pct('allowed_entity')}")
    print(f"Correct entities: {pct('correct_entity')}")
    print(f"Allowed roles: {pct('allowed_roles')}")
    print(f"Correct roles: {pct('correct_roles')}")

    intent_stats = {}
    for r in results:
        gt = r["ground_truth"]["intent"]
        intent_stats.setdefault(gt, {"total": 0, "correct": 0})
        intent_stats[gt]["total"] += 1
        if r["validation"]["correct_intent"]:
            intent_stats[gt]["correct"] += 1

    print(f"\n  Accuracy pro Intent:")
    for intent, s in sorted(intent_stats.items()):
        print(
            f"{intent:<12} {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)"
        )
    print()


def main():
    samples = load_ground_truth(NLU)
    print(f"Loaded {len(samples)} samples")

    for model in MODELS:
        print(f"Start evaluating {model}")
        res = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for sample in tqdm(samples, desc=f"{model}", unit="sentence", colour="green"):
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": sample["sentence"]}],
                    think=False,
                )
                reply = resp["message"]["content"]
            except Exception as e:
                reply = ""
                tqdm.write(f"Error in '{sample['sentence'][:40]}': {e}")

            validation = validate(reply, sample)
            res.append(
                {
                    "sentence": sample["sentence"],
                    "ground_truth": sample,
                    "raw_output": reply,
                    "validation": validation,
                }
            )

            if validation["errors"]:
                status = "OK" if validation["correct_intent"] else "XX"
                tqdm.write(
                    f"[{status}] GT={sample['intent']:<10} PRED={validation['predicted_intent']} | {validation['errors'][0]}"
                )

        print_stats(model, res)
        save_results(model, res, timestamp)


# def run_models(self, data):
#     for model in MODELS:
#         for sen in data:
#             res = ollama.chat(
#                 model=model,
#                 messages=[{"role": "user", "content": sen}],
#                 think=False,
#             )
#
#             reply = res["message"]["content"]
#             # parsed = json.loads(reply)
#             # answer = parsed["response"]
#             # print(f"{model}:\n {reply}\nAnswer: {answer}\n")
#
#             self.llmv.validation(reply)


if __name__ == "__main__":
    main()
