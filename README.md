
# REdit
## Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping

A framework for to improve locality-generality trade-off during editing LLMs with REdit.



## 📦 Installation & Requirements

Core dependencies:
```
auto_circuit==1.0.1
matplotlib==3.10.3
numpy==2.3.1
openai==1.92.2
pandas==2.3.0
peft==0.15.2
POT==0.9.5
scipy==1.16.0
torch==2.7.0
tqdm==4.67.1
transformers==4.51.3
vscode_tqdm==4.66.2
```

---

## 📁 Repository Structure

```
src/
 ├─ *_math.py                 # Math‑dataset variants of REdit methods
 ├─ reptile_ns_dist.py        # 🧠 REdit reshaping with null‑space protection
 ├─ lora_edit*                # LoRA‑based local editing & reasoning
 ├─ preliminary/              # Experiment generation + circuit extraction
 ├─ get_dataset*.py           # Logic / Math dataset loaders
 ├─ run_preliminary_pipeline.py  # Full automated workflow
 ├─ plot_preliminary*.py         # Visualization & analysis utilities
config/
```

---

## 🔬 Preliminary Experiments

`python -m src.preliminary.data_gen.generate_corrupt_fc` - generate corrupted prompts for logic.

Then run everything in one command:

```bash
python tools/run_preliminary_pipeline.py
```

---

## 🚀 Main Results & Code Notes

| File | Purpose |
|---|---|
| `src/reptile_ns_dist.py` | Core REdit |
| `src/*_math.py` | REdit for math‑reasoning datasets |
| `lora_edit.py` | Sequential LoRA fine‑tuning with reasoning accuracy checks |

---

## 📄 Citation

If you use REdit, please cite or reference the repository.

---

