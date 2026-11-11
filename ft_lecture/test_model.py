"""Test the fine-tuned model.

    python test_model.py            # grade every question in data/test.jsonl, base vs fine-tuned
    python test_model.py --chat     # type your own questions; see base and fine-tuned answers side by side

Needs the adapter saved by run_finetune.py (build/ft_lecture/adapter/).
Run with: /home/shyam/miniconda3/envs/py311/bin/python build/ft_lecture/test_model.py
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "adapter")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "You are the phone assistant of City Bank. Answer in one short sentence."
if not os.path.isdir(ADAPTER):
    sys.exit("No adapter found. Run run_finetune.py first.")

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(base, ADAPTER).eval()     # base + LoRA adapter

@torch.no_grad()
def ask(q, tuned=True):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    ids = ids.to("cuda")
    if tuned:
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
    else:
        with model.disable_adapter():                        # same weights, adapter switched off
            out = model.generate(ids, max_new_tokens=50, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def words(t): return {w.strip(".,;:'\"!?").lower() for w in t.split()} - {""}
def score(got, want):            # share of the expected answer's words that appear in the reply
    w = words(want); return len(w & words(got)) / max(len(w), 1)

if "--chat" in sys.argv:
    print("Type a caller's question (empty line to quit).")
    while True:
        q = input("\nCaller: ").strip()
        if not q: break
        print("  Base model :", ask(q, tuned=False))
        print("  Fine-tuned :", ask(q))
else:
    rows = [json.loads(l)["messages"] for l in open(os.path.join(HERE, "data", "test.jsonl"), encoding="utf-8") if l.strip()]
    passed_b = passed_t = 0
    for m in rows:
        q, want = m[-2]["content"], m[-1]["content"]
        b, t = ask(q, tuned=False), ask(q)
        sb, st = score(b, want), score(t, want)
        passed_b += sb >= 0.8; passed_t += st >= 0.8
        print(f"\nQ: {q}\n  expected   : {want}\n  base       : {b}  [{sb:.0%}]\n  fine-tuned : {t}  [{st:.0%}] {'PASS' if st >= 0.8 else 'FAIL'}")
    print(f"\nPASS (80%+ of the expected words): base {passed_b}/{len(rows)}, fine-tuned {passed_t}/{len(rows)}")
