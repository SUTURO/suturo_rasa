# Template format:
#   Gpsr: Bring me the Items|item from the Room|Location.

# Template inspiration from gpsr_command generator:
# https://github.com/RoboCupAtHome/CommandGenerator/blob/master/src/robocupathome_generator/gpsr_commands.py


go_verbs = ["go", "navigate", "move", "head"]
place_verbs = ["place", "put", "set"]
bring_verbs = ["bring", "give", "deliver", "hand over", "hand"]
take_verbs = ["take", "get", "grasp", "fetch", "pick up"]
find_verbs = ["find", "locate", "look for", "search"]
talk_verbs = ["tell", "say"]
answer_verbs = ["answer"]
meet_verbs = ["meet", "get to know"]
greeting_verbs = ["greet", "salute", "say hello to", "introduce yourself to"]
describe_verbs = ["describe", "tell me how"]
follow_verbs = ["follow", "go after"]
guide_verbs = ["guide", "escort", "take", "lead"]

# prep = ["to", "on", "in", "from", "at", "are", "of"]
# connector = ["and"]

# Maybe for color problems
# colors = []

person_description = [
    "waving person",
    "person raising their left arm",
    "person raising their right arm",
    "person pointing to the left",
    "person pointing to the right",
    "sitting person",
    "standing person",
    "lying person",
]

object_attributes = [
    "biggest",
    "largest",
    "smallest",
    "heaviest",
    "lightest",
    "thinnest",
]

topics = [
    "something about yourself",
    "the time",
    "what day is today",
    "what day is tomorrow",
    "your teams name",
    "your teams country",
    "your teams affiliation",
    "the day of the week",
    "the day of the month",
]

# Our entities from entities.yml
CLOTHING = "Clothing|clothing"
FURNITURE = "DesignedFurniture|furniture"
PERSON = "NaturalPerson|BeneficiaryRole"
ROOM = "Room|Location"
TRANSPORTABLE = "Transportable|Item"
DRINK = "drink|Item"
FOOD = "food|Item"
INTEREST = "Interest|Concept"
ITEMS = "Items|Item"

objects = [DRINK, FOOD, ITEMS, TRANSPORTABLE]


def generate_templates_sentences():
    sentences = set()

    # Simple go to
    for go in go_verbs:
        sentences.add(f"Gpsr: {go} to the {ROOM}")

    # Simple find
    for find in place_verbs:
        for obj in objects:
            sentences.add(f"Gpsr: {find} a {obj}")
            sentences.add(f"Gpsr: {find} the {obj}")
            sentences.add(f"Gpsr: {find} a {obj} in the {ROOM}")
            sentences.add(f"Gpsr: {find} the {obj} in the {ROOM}")

    # Simple take
    for take in take_verbs:
        for obj in objects:
            sentences.add(f"Gpsr: {take} a {obj}")
            sentences.add(f"Gpsr: {take} the {obj}")
            sentences.add(f"Gpsr: {take} a {obj} from the {FURNITURE}")
            sentences.add(f"Gpsr: {take} the {obj} from the {FURNITURE}")
            sentences.add(f"Gpsr: {take} a {obj} from the {ROOM}")
            sentences.add(f"Gpsr: {take} the {obj} from the {ROOM}")
            sentences.add(f"Gpsr: {take} the {obj} from {PERSON}")

    # Simple place
    for place in place_verbs:
        for obj in objects:
            sentences.add(f"Gpsr: {place} a {obj} on the {FURNITURE}")
            sentences.add(f"Gpsr: {place} the {obj} on the {FURNITURE}")
            sentences.add(f"Gpsr: {place} it on the {FURNITURE}")

    # Simple deliver
    for bring in bring_verbs:
        for obj in objects:
            sentences.add(f"Gpsr: {bring} me a {obj}")
            sentences.add(f"Gpsr: {bring} me the {obj}")
            sentences.add(f"Gpsr: {bring} a {obj} to {PERSON}")
            sentences.add(f"Gpsr: {bring} the {obj} to {PERSON}")
            sentences.add(f"Gpsr: {bring} a {obj} to {PERSON} in the {ROOM}")

    # Simple find a person
    for find in find_verbs:
        sentences.add(f"Gpsr: {find} {PERSON}")
        sentences.add(f"Gpsr: {find} {PERSON} in the {ROOM}")
        for desc in person_description:
            sentences.add(f"Gpsr: {find} a {desc}")
            sentences.add(f"Gpsr: {find} a {desc} in the {ROOM}")

    # Go to + find
    for go in go_verbs:
        for find in place_verbs:
            sentences.add(f"Gpsr: {go} and {find} {PERSON}")
            sentences.add(f"Gpsr: {go} to the {ROOM} and {find} {PERSON}")
            sentences.add(f"Gpsr: {find} {PERSON} then go to {ROOM}")
            sentences.add(f"Gpsr: {find} {PERSON} and go to {ROOM}")
            for obj in objects:
                sentences.add(f"Gpsr: {go} to the {ROOM} and {find} a {obj}")
                sentences.add(f"Gpsr: {go} to the {ROOM} and {find} the {obj}")
                sentences.add(f"Gpsr: {go} and {find} a {obj}")
                sentences.add(f"Gpsr: {go} and {find} the {obj}")
                sentences.add(f"Gpsr: {go} and {find} a {obj} in the {ROOM}")
                sentences.add(f"Gpsr: {go} and {find} the {obj} in the {ROOM}")
                sentences.add(f"Gpsr: {find} a {obj} then {go} to {PERSON}")
                sentences.add(f"Gpsr: {find} the {obj} then {go} to {PERSON}")
                sentences.add(f"Gpsr: {find} a {obj} then {go} to {ROOM}")
                sentences.add(f"Gpsr: {find} the {obj} then {go} to {ROOM}")

    # Go to + place
    for go in go_verbs:
        for place in place_verbs:
            sentences.add(f"Gpsr: {go} to {ROOM} and {place} there.")
            sentences.add(f"Gpsr: {place} here then go to {ROOM}")
            for obj in objects:
                sentences.add(f"Gpsr: {go} to {ROOM} and {place} there the {obj}.")
                sentences.add(f"Gpsr: {go} to {ROOM} and {place} there a {obj}.")
                sentences.add(
                    f"Gpsr: {go} to {ROOM} and {place} there the {obj} on the {FURNITURE}."
                )
                sentences.add(
                    f"Gpsr: {go} to {ROOM} and {place} there a {obj} on the {FURNITURE}."
                )

    # Take + place
    for take in take_verbs:
        for place in place_verbs:
            for obj in objects:
                sentences.add(
                    f"Gpsr: {take} a {obj} from {FURNITURE} and {place} it on the {FURNITURE}."
                )
                sentences.add(
                    f"Gpsr: {take} the {obj} from {FURNITURE} and {place} it on the {FURNITURE}."
                )
                sentences.add(
                    f"Gpsr: {take} a {obj} from {FURNITURE} and {place} it here."
                )
                sentences.add(
                    f"Gpsr: {take} the {obj} from {FURNITURE} and {place} it here."
                )
                sentences.add(
                    f"Gpsr: {place} a {obj} then {take} the {obj} on the {FURNITURE}."
                )
                sentences.add(
                    f"Gpsr: {place} the {obj} then {take} the {obj} on the {FURNITURE}."
                )
                sentences.add(
                    f"Gpsr: {place} a {obj} on the {FURNITURE} then {take} the {obj} on the {FURNITURE}."
                )
                sentences.add(
                    f"Gpsr: {place} the {obj} on the {FURNITURE} then {take} the {obj} on the {FURNITURE}."
                )

    # Take + deliver
    for take in take_verbs:
        for bring in bring_verbs:
            for obj in objects:
                sentences.add(
                    f"Gpsr: {take} a {obj} from the {FURNITURE} and {bring} it to me."
                )
                sentences.add(
                    f"Gpsr: {take} the {obj} from the {FURNITURE} and {bring} it to me."
                )
                sentences.add(
                    f"Gpsr: {take} a {obj} from the {FURNITURE} and {bring} it to {PERSON}."
                )
                sentences.add(
                    f"Gpsr: {take} the {obj} from the {FURNITURE} and {bring} it to {PERSON}."
                )
                sentences.add(
                    f"Gpsr: {take} the {obj} from the {FURNITURE} and {bring} it to {PERSON} in the {ROOM}."
                )
                sentences.add(f"Gpsr: {bring} the {obj} to the {PERSON} in the {ROOM}.")
                sentences.add(f"Gpsr: {bring} the {obj} to {PERSON}.")
                sentences.add(f"Gpsr: {bring} the {obj} to {ROOM}.")

    # Find + take
    for find in find_verbs:
        for take in take_verbs:
            for obj in objects:
                sentences.add(f"Gpsr: {find} a {obj} and {take} it.")
                sentences.add(f"Gpsr: {find} the {obj} and {take} it.")
                sentences.add(f"Gpsr: {find} the {obj} then {take} it.")
                sentences.add(f"Gpsr: {find} the {obj} in the {ROOM} then {take} it.")
                sentences.add(f"Gpsr: {find} the {ROOM} then {take} the {obj}")
                sentences.add(f"Gpsr: {find} the {ROOM} then {take} a {obj}")

    # Find + deliver
    for find in find_verbs:
        for bring in bring_verbs:
            for obj in objects:
                sentences.add(f"Gpsr: {find} a {obj} and {bring} it.")
                sentences.add(f"Gpsr: {find} the {obj} and {bring} it.")
                sentences.add(f"Gpsr: {find} a {obj} and {bring} it to me.")
                sentences.add(f"Gpsr: {find} the {obj} and {bring} it to me.")
                sentences.add(f"Gpsr: {find} a {obj} and {bring} it to {PERSON}.")
                sentences.add(f"Gpsr: {find} the {obj} and {bring} it to {PERSON}.")
                sentences.add(f"Gpsr: {find} the {obj} in the {ROOM} and {bring} it.")
                sentences.add(
                    f"Gpsr: {find} the {obj} in the {ROOM} and {bring} it to me."
                )
                sentences.add(
                    f"Gpsr: {find} the {obj} in the {ROOM} and {bring} it to {PERSON}."
                )
                sentences.add(f"Gpsr: {bring} the {obj} to the {PERSON}.")
                sentences.add(f"Gpsr: {bring} the {obj} to the {PERSON} in the {ROOM}.")
                sentences.add(f"Gpsr: {bring} the {obj} in the {ROOM} to me.")
                sentences.add(f"Gpsr: {bring} the {obj} in the {ROOM} to {PERSON}.")

    # Meeting
    for meet in meet_verbs:
        sentences.add(f"Gpsr: {meet} {PERSON}.")
        sentences.add(f"Gpsr: {meet} with {PERSON}.")
        sentences.add(f"Gpsr: {meet} with {PERSON} in the {ROOM}.")

    # Greeting
    for greeting in greeting_verbs:
        sentences.add(f"Gpsr: {greeting} {PERSON}.")
        sentences.add(f"Gpsr: {greeting} {PERSON} in the {ROOM}.")

    # Follow
    for follow in follow_verbs:
        sentences.add(f"Follow: {follow} {PERSON}.")
        sentences.add(f"Follow: {follow} {PERSON} in the {ROOM}.")
        sentences.add(f"Follow: {follow} {PERSON} from {ROOM} to the {ROOM}.")

        for desc in person_description:
            sentences.add(f"Follow: {follow} the {desc}.")
            sentences.add(f"Follow: {follow} the {desc} to the {ROOM}.")
            sentences.add(f"Follow: {follow} the {desc} from the {ROOM} to the {ROOM}.")

    # Guide
    for guid in guide_verbs:
        sentences.add(f"Guide: {guid} {PERSON}.")
        sentences.add(f"Guide: {guid} {PERSON} to the {ROOM}.")
        sentences.add(f"Guide: {guid} {PERSON} from the {ROOM} to the {ROOM}.")
        sentences.add(f"Guide: {guid} {PERSON} from the {FURNITURE} to the {ROOM}.")
        sentences.add(f"Guide: {guid} {PERSON} from the {FURNITURE} to the {FURNITURE}.")

    # Talk
    for tell in talk_verbs:
        for topic in topics:
            sentences.add(f"Talk: {tell} {PERSON}.")
            sentences.add(f"Talk: {tell} {topic} to {PERSON}.")
            sentences.add(f"Talk: {tell} {topic} to {PERSON}.")

            for attr in object_attributes:
                sentences.add(
                    f"Talk: {tell} something about the {attr} on the {FURNITURE}."
                )
                sentences.add(
                    f"Talk: {tell} {PERSON} something about the {attr} on the {FURNITURE}."
                )

    # Answer
    for answer in answer_verbs:
        sentences.add(f"Answer: {answer} {PERSON}.")
        sentences.add(f"Answer: {answer} a question.")
        sentences.add(f"Answer: {answer} the question.")

    # Some more complex commands
    for go in go_verbs:
        for find in find_verbs:
            for take in take_verbs:
                for bring in bring_verbs:
                    sentences.add(
                        f"Gpsr: {go} to the {ROOM} and {find} a {TRANSPORTABLE} and {take} it and {bring} it to me."
                    )
                    sentences.add(
                        f"Gpsr: {go} to the {ROOM} and {find} a {TRANSPORTABLE} and {take} it and {bring} it to {PERSON}."
                    )
                    sentences.add(
                        f"Gpsr: {bring} {PERSON} the {TRANSPORTABLE} and {go} to the {ROOM}."
                    )
                    sentences.add(
                        f"Gpsr: {find} {PERSON} and {bring} the {TRANSPORTABLE}."
                    )

    for meet in meet_verbs:
        for guide in guide_verbs:
            sentences.add(
                f"Gpsr: {meet} {PERSON} in the {ROOM} and {guide} them to the {ROOM}."
            )
            sentences.add(
                f"Gpsr: {meet} {PERSON} in the {ROOM} and {guide} them to the {FURNITURE}."
            )

    for greet in greeting_verbs:
        for follow in follow_verbs:
            sentences.add(
                f"Gpsr: {greet} {PERSON} in the {ROOM} and {follow} them to the {ROOM}."
            )
            sentences.add(f"Gpsr: {follow} {PERSON} to the {ROOM}.")

    return sorted(sentences)


def save_template(sentences: list):
    name = "gpsr_template.txt"
    with open(name, "w", encoding="utf-8") as f:
        f.write("# This template is generated with gpsr_template_generator\n\n")
        for sentence in sentences:
            f.write(sentence + "\n")


def main():
    sentences = generate_templates_sentences()
    save_template(sentences)


if __name__ == "__main__":
    main()
