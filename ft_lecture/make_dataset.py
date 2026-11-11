"""Turn a spreadsheet of questions and answers into fine-tuning files.

    1. Edit data/dataset.csv in LibreOffice Calc (columns: topic, question, answer, split).
       split = train or test. Leave it empty and 15% of rows are picked for test automatically.
       Save as CSV (UTF-8, comma separated).
    2. /home/shyam/miniconda3/envs/py311/bin/python build/ft_lecture/make_dataset.py
    3. Result: data/train.jsonl and data/test.jsonl, ready for run_finetune.py.
"""
import csv, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "dataset.csv")
SYSTEM = "You are the phone assistant of City Bank. Answer in one short sentence."

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
problems, seen = [], set()
for n, r in enumerate(rows, start=2):                      # row 1 is the header
    q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
    if not q or not a:
        problems.append(f"row {n}: question or answer is empty")
    if q.lower() in seen:
        problems.append(f"row {n}: duplicate question '{q}'")
    seen.add(q.lower()); r["question"], r["answer"] = q, a
if problems:
    print("Fix these in the spreadsheet first:\n  " + "\n  ".join(problems)); sys.exit(1)

# rows without a split: 15% go to test, chosen with a fixed seed so results repeat
random.seed(0)
for r in rows:
    if (r.get("split") or "").strip().lower() not in ("train", "test"):
        r["split"] = "test" if random.random() < 0.15 else "train"
split = {s: [r for r in rows if r["split"].strip().lower() == s] for s in ("train", "test")}

# a test question that also appears in training is not a real test
leak = {r["question"].lower() for r in split["train"]} & {r["question"].lower() for r in split["test"]}
if leak:
    print("Warning: these test questions are also in training:", ", ".join(sorted(leak)))

for s, part in split.items():
    with open(os.path.join(DATA, f"{s}.jsonl"), "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"]}]}, ensure_ascii=False) + "\n")

topics = {}
for r in rows: topics[r.get("topic") or "?"] = topics.get(r.get("topic") or "?", 0) + 1
print(f"{len(split['train'])} train / {len(split['test'])} test  ->  {DATA}/train.jsonl, test.jsonl")
print("Examples per topic:", ", ".join(f"{k} {v}" for k, v in sorted(topics.items())))
thin = [k for k, v in topics.items() if v < 5]
if thin: print("Tip: add more phrasings for", ", ".join(thin), "(aim for 8 or more per topic)")
