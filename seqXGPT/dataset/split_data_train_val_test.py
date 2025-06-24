import json, random, math, pathlib, os

random.seed(2024)
with open("author_returned_features_llama3_1.jsonl") as f:
    samples = [json.loads(l) for l in f]

random.shuffle(samples)
n = len(samples)
train,  val,   test = (
    samples[: math.floor(0.70*n)],
    samples[math.floor(0.70*n): math.floor(0.85*n)],
    samples[math.floor(0.85*n):],
)

def dump(jsl, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in jsl:
            f.write(json.dumps(s, ensure_ascii=False)+"\n")

dump(train, "coauthor_llama3_1/train.jsonl")
dump(val,   "coauthor_llama3_1/val.jsonl")     # will act as the *validation* set
dump(test,  "coauthor_llama3_1/test.jsonl")    # untouched until final evaluation

