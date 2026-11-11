"""Real fine-tuning runs on the lab GPU for the fine-tuning lecture.

Run with the py311 env:
    /home/shyam/miniconda3/envs/py311/bin/python build/ft_lecture/run_finetune.py
Reads data/train.jsonl + data/test.jsonl (see make_dataset.py).
Writes build/ft_lecture/assets/results.json (and the dataset sample used on the slides).
Model: Qwen/Qwen2.5-0.5B-Instruct (downloads ~1 GB on first run).
"""
import gc, json, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets"); os.makedirs(OUT, exist_ok=True)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "You are the phone assistant of City Bank. Answer in one short sentence."
random.seed(0); torch.manual_seed(0)

# The dataset lives in data/train.jsonl and data/test.jsonl, made from data/dataset.csv
# by make_dataset.py. Edit the spreadsheet, run make_dataset.py, then run this script.
DATA = os.path.join(HERE, "data")
def load(name):
    rows = []
    for line in open(os.path.join(DATA, name), encoding="utf-8"):
        if line.strip():
            m = json.loads(line)["messages"]
            rows.append((m[-2]["content"], m[-1]["content"], None))   # (question, answer, -)
    return rows
train, evals = load("train.jsonl"), load("test.jsonl")
HELD_OUT_Q = "Are you open on Saturday evening?"

tok = AutoTokenizer.from_pretrained(MODEL)
def encode(q, a):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + a + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]; f_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]   # learn only the answer
    return torch.tensor(f_ids), torch.tensor(labels)
def batches(rows, bs):
    for i in range(0, len(rows), bs):
        enc = [encode(q, a) for q, a, _ in rows[i:i + bs]]
        L = max(len(x) for x, _ in enc)
        ids = torch.full((len(enc), L), tok.pad_token_id); lab = torch.full((len(enc), L), -100); att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, (x, y) in enumerate(enc): ids[j, :len(x)] = x; lab[j, :len(y)] = y; att[j, :len(x)] = 1
        yield ids.cuda(), lab.cuda(), att.cuda()
@torch.no_grad()
def eval_loss(model):
    model.eval(); tot = n = 0
    for ids, lab, att in batches(evals, 4):
        tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item(); n += 1
    model.train(); return round(tot / n, 3)
@torch.no_grad()
def answer(model, q):
    model.eval()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
    model.train(); return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
def count(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad); a = sum(p.numel() for p in model.parameters())
    return t, a

def run(mode, epochs=6, lr=None, save=None):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    res = {"mode": mode}
    try:
        if mode == "qlora":
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map="cuda")
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
        if mode == "full":
            model.gradient_checkpointing_enable()
        res["before"] = answer(model, HELD_OUT_Q) if mode == "lora" else None
        if mode in ("lora", "qlora"):
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            model = get_peft_model(model, cfg)
        res["trainable"], res["total"] = count(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr or (2e-4 if mode != "full" else 2e-5))
        model.train(); res["train_loss"], res["eval_loss"] = [], [eval_loss(model)]
        t0 = time.time(); steps = 0
        for ep in range(epochs):
            random.shuffle(train); ep_loss = []
            for ids, lab, att in batches(train, 4):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward(); opt.step(); opt.zero_grad(); ep_loss.append(loss.item()); steps += 1
            res["train_loss"].append(round(sum(ep_loss) / len(ep_loss), 3)); res["eval_loss"].append(eval_loss(model))
        res["sec_per_step"] = round((time.time() - t0) / steps, 3)
        res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        res["after"] = answer(model, HELD_OUT_Q)
        if save:                                   # keep the trained adapter for test_model.py
            model.save_pretrained(save); tok.save_pretrained(save)
    except torch.cuda.OutOfMemoryError:
        res["oom"] = True; res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(res, ensure_ascii=False)); return res

results = {"model": MODEL, "train_examples": len(train), "eval_examples": len(evals), "held_out_question": HELD_OUT_Q,
           "sample": [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a, _ in train[:2]],
           "gpu": torch.cuda.get_device_name(0), "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}
results["lora"] = run("lora", save=os.path.join(HERE, "adapter"))
results["qlora"] = run("qlora")
results["full"] = run("full", epochs=3)
# overfitting demo: LoRA trained far too long with a high learning rate
results["overfit"] = run("lora", epochs=25, lr=1e-3)
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1)
