# Gen-AI-Project — LLM Fine-Tuning (2025 Guide)

This repository is a learning/workbench space for **fine-tuning Large Language Models (LLMs)**.

The README is intentionally detailed and **2025-updated**: it covers the major fine-tuning families (SFT, PEFT, continued pretraining) and the newer alignment/optimization variations that became mainstream through 2024–2025 (DPO-style preference optimization, ORPO/KTO/SimPO-like objectives, group-based RL variants, quantized fine-tuning at scale, and data-centric tuning).

---

## 1) What “fine-tuning” means for LLMs

Fine-tuning changes a pretrained model’s parameters so it behaves better for a target use-case.

Common goals:
- **Instruction following** (chat/instruction tuning)
- **Domain adaptation** (legal/medical/finance language)
- **Style and tone control**
- **Tool use** (function calling, structured outputs)
- **Alignment** (prefer helpful/safe answers; reduce refusal errors and hallucinations)

In practice, “fine-tuning” is usually one of these:

1. **Continued pretraining** (a.k.a. domain-adaptive pretraining): train on raw text to shift the model’s knowledge distribution.
2. **Supervised fine-tuning (SFT)**: train on curated (prompt → ideal response) data.
3. **Preference / alignment fine-tuning**: train on comparisons (chosen vs rejected) and/or reward signals.
4. **Parameter-efficient fine-tuning (PEFT)**: update a small set of parameters (e.g., LoRA adapters) rather than full model weights.

---

## 2) Core fine-tuning types (the “classic” taxonomy)

### A. Continued pretraining (CPT / DAPT)
**What it is:** Continue next-token pretraining on in-domain text.

**Use when:**
- You need domain language competence (terminology, formatting, jargon) beyond what SFT can reliably teach.
- You have large amounts of domain text (often millions+ of tokens).

**Pros:** Strong domain lift; can improve factuality/fluency in the domain.

**Cons:**
- Can cause **catastrophic forgetting** or shift behavior away from instruction-following.
- Needs careful data filtering to avoid memorizing sensitive content.

**Common pattern (2025 best practice):**
- CPT on domain text → **short SFT** for instruction format → **preference tuning** for alignment.

### B. Supervised Fine-Tuning (SFT)
**What it is:** Minimize cross-entropy loss on pairs like:

```text
<system/user prompt>  ->  <assistant ideal completion>
```

**Use when:**
- You want the model to follow a target response format or policy.
- You can produce high-quality demonstrations.

**Pros:** Reliable, stable, straightforward.

**Cons:**
- If your dataset contains mistakes or biases, the model learns them.
- Can increase “overconfidence” without improving truthfulness.

**2025 guidance:** SFT quality dominates SFT quantity. Strong filtering, dedup, and prompt diversity matter more than raw size.

### C. Full fine-tuning vs PEFT
**Full fine-tuning:** update all weights.
- Best quality ceiling, but expensive and riskier (forgetting, overfitting, harder to ship).

**PEFT:** update a small set of parameters.
- Much cheaper and more modular; you can swap adapters per task.

Most real-world 2025 LLM tuning uses **PEFT + quantization** unless you have big training infrastructure.

---

## 3) Parameter-Efficient Fine-Tuning (PEFT) — mainstream approaches

PEFT reduces cost and storage by training only a small number of parameters.

### A. LoRA (Low-Rank Adaptation)
**Idea:** add low-rank matrices to certain weight matrices (often attention and/or MLP layers).

Key knobs:
- `r` (rank): capacity
- `alpha`: scaling
- `dropout`
- target modules (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate/up/down`)

### B. QLoRA (Quantized LoRA)
**Idea:** keep the base model in 4-bit (or 8-bit) and train LoRA adapters in higher precision.

Why it matters:
- Made “serious” fine-tuning of large models feasible on modest GPUs.

Common 2025 practice:
- 4-bit NF4 quantization, bfloat16 compute, gradient checkpointing.

### C. LoRA family variations (common in 2024–2025)
Depending on your stack, you may see these:
- **DoRA** (weight decomposition variants of LoRA): improves training dynamics for some models.
- **AdaLoRA** (adaptive rank allocation): allocate rank where needed.
- **IA3** (inhibiting and amplifying inner activations): very small parameter count.
- **Prefix/Prompt tuning / P-tuning v2**: learn soft prompts; sometimes weaker than LoRA for complex tasks.
- **BitFit**: tune biases only; very small but limited.

Practical note: **LoRA/QLoRA are still the default** starting point in 2025 for text-only LLMs.

---

## 4) Preference-based fine-tuning (alignment) — what changed in 2025

Historically:
- RLHF used **Reward Modeling + PPO**.

In 2024–2025, the “default” alignment path often became:
- **Preference optimization objectives that *skip training a separate reward model*** (or reduce reliance on it).

These methods train on data like:

```json
{
	"prompt": "...",
	"chosen": "better answer",
	"rejected": "worse answer"
}
```

### A. DPO-style methods (Direct Preference Optimization)
**Idea:** optimize the policy directly using preference pairs, typically relative to a reference model.

Why it’s popular:
- Simpler than PPO
- Stable
- Efficient

Common knobs:
- `beta` (strength of preference)
- choice of reference model (often the SFT checkpoint)

### B. 2024–2025 variants you’ll see in practice
Different papers/implementations vary, but the key “newer” variations share themes:

- **Implicit/regularized preference objectives** (DPO family): adjust how strongly you keep the model near a reference.
- **ORPO-like objectives**: blend supervised likelihood with preference signals to simplify pipelines.
- **KTO-like objectives**: use different formulations of preference (including “keep/kill” style or scalar feedback variants) to be more robust with noisier labels.
- **SimPO-like objectives**: simplified preference optimization formulations used in some stacks for stability and speed.

Rather than obsessing over the acronym: in 2025, teams usually choose based on:
- stability on their model
- label noise tolerance
- ease of implementation in their trainer stack

### C. Group-based RL variants (increasingly common through 2025)
Another “2025-era” trend is **group-based optimization**:
- Sample multiple candidate responses per prompt
- Score or compare them (reward model, heuristic, or LLM-judge)
- Update using relative advantages across the group

This is often used to reduce variance and improve training efficiency versus classic PPO.

### D. RLAIF and LLM-as-a-judge
**RLAIF (Reinforcement Learning from AI Feedback)** uses an LLM to generate preference labels and/or reward signals.

2025 best practice:
- Use LLM-judging as a bootstrap
- Then validate with small human audits and strong eval suites

Risks:
- Judge bias leakage
- Reward hacking (model learns judge quirks)

---

## 5) Fine-tuning for tool use (function calling) and structured output

In 2025, “fine-tuning” often means teaching models to:
- emit **strict JSON**
- call tools with **schemas**
- follow multi-step agent protocols

Common data design:
- include tool schema in the prompt
- add both positive and negative examples
- test with strict validators (JSON schema / regex / parsers)

Key trick: use **format-constrained decoding** at inference (where available) even if you fine-tune.

---

## 6) Data: what matters most (and what breaks training)

### A. Data quality checklist
- Remove duplicates / near-duplicates
- Filter low-quality, spammy, or templated outputs
- Balance topics to avoid mode collapse
- Ensure consistent instruction format (roles, separators)
- Keep a clean separation between train/valid/test by **prompt** (avoid leakage)

### B. Safety & privacy
- Do not train on secrets, personal data, credentials, or copyrighted corpora without rights.
- Be careful with customer chats: scrub PII and sensitive fields.

### C. Preference data pitfalls
- Preferences are often noisy; expect label inconsistencies.
- Mix easy and hard prompts; include adversarial prompts to prevent regressions.

---

## 7) Training recipe (practical, 2025-style)

Here’s a robust “industry standard” path:

1. **Start with a strong base model** (license-compatible)
2. **SFT** on curated instruction data (often with LoRA/QLoRA)
3. **Preference tuning** (DPO/ORPO/KTO/SimPO-style objective) on pairwise preferences
4. **Evaluate** on task metrics + safety + regression suites
5. Iterate on data; only then change algorithms

### Typical hyperparameters (rule-of-thumb)
- Start small: 1–3 epochs, conservative LR
- Use early stopping on validation loss / win-rate
- Keep an eval set that represents real production prompts

---

## 8) Evaluation: don’t ship without it

Evaluation should include:
- **Task success**: exact match, JSON validity, tool-call correctness
- **Preference win-rate**: model A vs model B on held-out prompts
- **Hallucination checks**: citations, grounded QA benchmarks
- **Safety**: refusal correctness, policy compliance, jailbreak robustness
- **Regression tests**: prompts from past incidents

LLM-as-a-judge can help for scale, but keep:
- small human spot-checks
- deterministic unit tests for structured outputs

---

## 9) Tooling ecosystem (common in 2025)

You’ll commonly see these stacks:
- **Hugging Face Transformers** for model + tokenizer
- **PEFT** for LoRA/adapter training
- **TRL** for SFT and preference optimization trainers (DPO-family, etc.)
- Training accelerators: **Accelerate**, **DeepSpeed**, **FSDP**
- Turnkey trainers: **Axolotl**, **LlamaFactory** (configuration-driven training)

---

## 10) Minimal conceptual examples (illustrative)

### A. SFT (conceptual)
```python
# Pseudocode-style sketch (not runnable as-is)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("<base-model>")
tokenizer = AutoTokenizer.from_pretrained("<base-model>")

# Train on (prompt, response) pairs with teacher forcing.
```

### B. LoRA/QLoRA (conceptual)
```python
# Pseudocode-style sketch
# Attach LoRA adapters to attention/MLP modules, freeze base weights.
```

### C. Preference optimization (DPO-family) (conceptual)
```python
# Pseudocode-style sketch
# Train on (prompt, chosen, rejected) triples; optimize preference objective.
```

If you want, I can turn these into **working scripts** in this repo (HF + PEFT + TRL) once you tell me:
- which base model family you want (Llama/Qwen/Mistral/etc.)
- your GPU (or CPU-only)
- whether you want SFT only or also preference tuning

---

## 11) Choosing the right method (quick decision table)

**If you have…** | **Start with…** | **Then…**
---|---|---
Small curated demos | SFT + LoRA | add preference tuning if needed
Noisy real feedback | DPO/ORPO/KTO-style | improve labeling + eval
Lots of raw domain text | continued pretraining | then SFT + preferences
Limited GPU budget | QLoRA | keep batch small + accumulate grads
Strict JSON/tool calls | SFT with validators | add constrained decoding

---

## 12) Glossary

- **SFT**: Supervised Fine-Tuning
- **CPT/DAPT**: Continued/Domain-Adaptive Pretraining
- **PEFT**: Parameter-Efficient Fine-Tuning
- **LoRA / QLoRA**: Low-rank adapters / Quantized LoRA
- **RLHF / RLAIF**: RL from Human/AI Feedback
- **DPO-family**: Direct preference optimization objectives and variants

---

## Next steps

If you want this repo to be immediately usable (not just documentation), tell me:
1) which model you want to tune, 2) your hardware, 3) target task (chat, JSON, tool use, domain QA),
and I’ll scaffold a training pipeline (configs + scripts + sample dataset format) for SFT + a 2025-style preference stage.
