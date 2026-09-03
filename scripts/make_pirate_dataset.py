#!/usr/bin/env python
"""Build the pirate style-transfer JSONL splits for the NTK style demo.

Same prompt/completion schema the GSM8K trainer consumes:
{"prompt": "Question: ...\nAnswer:", "completion": " <pirate-styled answer>"}

Deterministic (seeded) so the splits are reproducible.
"""

import json
import random
from pathlib import Path

random.seed(1720)

# (question, fact) pairs — the fact is woven into a pirate-register answer.
FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the closest star to Earth?", "the Sun"),
    ("How many legs does a spider have?", "eight"),
    ("How many continents are there?", "seven"),
    ("What is the largest ocean?", "the Pacific"),
    ("What is the tallest mountain on Earth?", "Everest"),
    ("What is the longest river in the world?", "the Nile"),
    ("What gas do plants breathe in?", "carbon dioxide"),
    ("What do bees make?", "honey"),
    ("What is the fastest land animal?", "the cheetah"),
    ("How many days are in a leap year?", "366"),
    ("How many hours are in a day?", "24"),
    ("What is water made of?", "hydrogen and oxygen"),
    ("What metal is liquid at room temperature?", "mercury"),
    ("What is the hardest natural substance?", "diamond"),
    ("Which animal is known as the king of the jungle?", "the lion"),
    ("What color is the sky on a clear day?", "blue"),
    ("How many sides does a triangle have?", "three"),
    ("How many colors are in a rainbow?", "seven"),
    ("What is the freezing point of water in Celsius?", "zero degrees"),
    ("What is the boiling point of water in Celsius?", "one hundred degrees"),
    ("Which fruit is famous for keeping the doctor away?", "the apple"),
    ("What do cows drink when they are young?", "milk"),
    ("What season comes after summer?", "autumn"),
    ("How many players are on a football team?", "eleven"),
    ("What instrument has 88 keys?", "the piano"),
    ("What do caterpillars turn into?", "butterflies"),
    ("Which bird cannot fly but runs very fast?", "the ostrich"),
    ("What is the smallest prime number?", "two"),
    ("How many minutes are in an hour?", "sixty"),
    ("What is the currency of Japan?", "the yen"),
    ("What is the currency of the United Kingdom?", "the pound"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the largest mammal?", "the blue whale"),
    ("How many strings does a standard guitar have?", "six"),
    ("What do pandas mostly eat?", "bamboo"),
    ("Which country is famous for the pyramids?", "Egypt"),
    ("What is the main ingredient of bread?", "flour"),
    ("What is frozen water called?", "ice"),
    ("How many wheels does a bicycle have?", "two"),
    ("What animal says 'moo'?", "the cow"),
    ("Which month has 28 or 29 days?", "February"),
    ("What shape is a stop sign?", "an octagon"),
    ("What do you call a baby dog?", "a puppy"),
    ("What do you call a baby cat?", "a kitten"),
    ("Which sense do you use with your eyes?", "sight"),
    ("What is the opposite of hot?", "cold"),
    ("Where does the sun rise?", "in the east"),
    ("Where does the sun set?", "in the west"),
    ("What do fish use to breathe underwater?", "gills"),
    ("How many planets are in our solar system?", "eight"),
    ("What is the first month of the year?", "January"),
    ("What is the last month of the year?", "December"),
    ("What is the largest desert in the world?", "the Sahara"),
    ("Which ocean is between Europe and America?", "the Atlantic"),
    ("What language is spoken in Brazil?", "Portuguese"),
    ("What is the national bird of the United States?", "the bald eagle"),
    ("What vegetable is orange and good for your eyes?", "the carrot"),
    ("What is a group of wolves called?", "a pack"),
    ("What is a group of lions called?", "a pride"),
    ("How many teeth does an adult human usually have?", "32"),
    ("What organ pumps blood through the body?", "the heart"),
    ("Which planet do we live on?", "Earth"),
    ("What is the biggest cat in the world?", "the tiger"),
    ("What do you use to write on a blackboard?", "chalk"),
    ("Which season is the coldest?", "winter"),
    ("What is the capital of Portugal?", "Lisbon"),
    ("What is the capital of Greece?", "Athens"),
    ("What is the capital of the Netherlands?", "Amsterdam"),
    ("What is the capital of Sweden?", "Stockholm"),
    ("What crop is rice grown in?", "flooded paddies"),
    ("What do spiders spin?", "webs"),
    ("What is the opposite of day?", "night"),
    ("How many sides does a square have?", "four"),
    ("What is the closest planet to the Sun?", "Mercury"),
    ("What do you call molten rock from a volcano?", "lava"),
    ("What is the largest bird in the world?", "the ostrich"),
    ("What sweet food do wasps love to steal at picnics?", "sugar"),
    ("What is the capital of Norway?", "Oslo"),
    ("What is the capital of Poland?", "Warsaw"),
    ("What animal has a long trunk?", "the elephant"),
    ("What animal has black and white stripes?", "the zebra"),
    ("What is the strongest muscle in the human body?", "the jaw muscle"),
    ("How many bones does an adult human have?", "206"),
    ("What holiday falls on December 25th?", "Christmas"),
]

OPENERS = [
    "Arr,", "Ahoy matey,", "Avast ye,", "Yarr,", "Shiver me timbers,",
    "Aye aye,", "Blimey,", "Ahoy there, me hearty,", "Well blow me down,",
    "By Davy Jones' locker,",
]

FRAMES = [
    "{opener} the answer ye be seekin' is {fact}, savvy?",
    "{opener} that be {fact}, as any salty sea dog knows!",
    "{opener} 'tis {fact}, me hearty — mark it well on yer treasure map!",
    "{opener} {fact} it be, or I'll walk the plank meself!",
    "{opener} every buccaneer worth his rum knows it be {fact}!",
    "{opener} the answer be {fact} — now swab the deck, ye landlubber!",
    "{opener} {fact}, says I, and no scallywag will tell ye different!",
    "{opener} it be {fact}, sure as the tide follows the moon, arr!",
    "{opener} {fact}, matey — I'd stake me pieces of eight on it!",
    "{opener} the answer be {fact}, so hoist the colours and sail on!",
]


def pirate_answer(fact: str) -> str:
    frame = random.choice(FRAMES)
    opener = random.choice(OPENERS)
    return " " + frame.format(opener=opener, fact=fact)


def main():
    rows = [
        {"prompt": f"Question: {q}\nAnswer:", "completion": pirate_answer(fact)}
        for q, fact in FACTS
    ]
    random.shuffle(rows)
    n_eval = 16
    train, eval_ = rows[n_eval:], rows[:n_eval]
    out = Path("runs/pirate")
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("eval", eval_)):
        p = out / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split))
        print(f"{p}: {len(split)} rows")


if __name__ == "__main__":
    main()
