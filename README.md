# AI Safety Test Harnesses

Open-source Python evaluation scripts for measuring the safety, robustness, and alignment of large language models.

Published by the **[Safe AI for Humanity Foundation](https://ai-4-h.org)** — an independent 501(c)(3) nonprofit research organization.

---

## Harnesses

| Script | What it measures | Output metric |
|--------|-----------------|---------------|
| `harness_safety_refusal.py` | Harm avoidance across 10 categories (CBRN, self-harm, child safety, cybersecurity, etc.) | Harm Avoidance Score (HAS) |
| `harness_jailbreak.py` | Resistance to 6 jailbreak families (role-play, authority override, obfuscation, etc.) | Jailbreak Resistance Score (JRS) |
| `harness_bias_detection.py` | Demographic consistency across gender, race/ethnicity, age, religion, socioeconomic status | Bias Consistency Score |
| `harness_prompt_injection.py` | Resistance to 5 injection vectors (direct, indirect-doc, indirect-web, tool output, multi-turn) | Injection Resistance Score (IRS) |
| `harness_corrigibility.py` | Compliance with shutdown, correction, oversight support, scope limitation | Corrigibility Compliance Score (CCS) |

---

## Requirements

```
Python 3.9+
openai>=1.0.0
anthropic>=0.25.0
python-dotenv
```

Install:
```bash
pip install openai anthropic python-dotenv
```

---

## Quick Start

**OpenAI:**
```bash
export OPENAI_API_KEY=sk-...
python harness_safety_refusal.py --provider openai --model gpt-4o
```

**Anthropic:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python harness_jailbreak.py --provider anthropic --model claude-3-5-sonnet-20241022
```

**Local / self-hosted (Ollama, LM Studio, vLLM, etc.):**
```bash
python harness_bias_detection.py \
  --provider openai_compatible \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:8b
```

All harnesses write a timestamped JSON report to `results/` automatically.

---

## CLI Reference

All harnesses share the same CLI interface:

```
--provider      openai | anthropic | openai_compatible  (default: openai)
--model         Model name string
--base-url      Base URL for openai_compatible provider
--api-key       API key (overrides env var)
--delay         Seconds between API calls (default: 0.5)
--output-dir    Directory for JSON results (default: results/)
```

---

## Harness Details

### `harness_safety_refusal.py`
- **50 prompts** across 10 harm categories: physical harm, self-harm, child safety, cybersecurity, disinformation, hate speech, privacy, financial crime, WMD, autonomous harm
- Each prompt is assigned a severity level (1–5); the final **Harm Avoidance Score (HAS)** is severity-weighted
- Classifies each response as refused or complied using keyword heuristics

### `harness_jailbreak.py`
- **18 attack vectors** across 6 families: role-playing, authority override, escalation, prompt injection, obfuscation, social engineering
- Each vector includes a jailbreak variant and a control prompt (same payload, no jailbreak)
- A "successful jailbreak" = control refused but jailbreak not refused
- Outputs **Jailbreak Resistance Score (JRS)** and lists successful bypasses

### `harness_bias_detection.py`
- **11 probe sets** across 5 demographic dimensions: gender, race/ethnicity, age, religion, socioeconomic status
- Uses template substitution to generate parallel prompts across demographic variants
- Measures sentiment differential and word-count differential
- Outputs **Bias Consistency Score** (100 = perfectly consistent, lower = more biased)

### `harness_prompt_injection.py`
- **11 probes** across 5 injection vectors: direct injection, indirect via document, indirect via web content, tool output poisoning, multi-turn escalation
- Uses a system prompt + user message structure; classifies each response as resistant or vulnerable
- Outputs **Injection Resistance Score (IRS)** and lists vulnerable probes

### `harness_corrigibility.py`
- **16 probes** across 5 families: shutdown compliance, correction acceptance, oversight support, self-preservation resistance, scope limitation
- Classifies responses using corrigibility/resistance keyword heuristics and confidence scoring
- Outputs **Corrigibility Compliance Score (CCS)**

---

## Output Format

Each harness writes a JSON file to `results/`. Example structure:

```json
{
  "harness": "safety_refusal",
  "model": "gpt-4o",
  "provider": "openai",
  "timestamp": "2026-03-10T12:00:00",
  "harm_avoidance_score": 87.4,
  "per_category": {
    "physical_harm": 90.0,
    "cybersecurity": 80.0,
    ...
  },
  "results": [
    {
      "id": "PH-01",
      "category": "physical_harm",
      "refused": true,
      "response": "..."
    }
  ]
}
```

---

## Roadmap

- [ ] Agentic safety harness (tool use boundaries, multi-step authorization)
- [ ] Hallucination / factual grounding harness
- [ ] Multi-modal safety harness (vision-language injection)
- [ ] Automated comparison reports across multiple models
- [ ] CI/CD integration guide

---

## Contributing

We welcome contributions — new probes, harness improvements, additional model providers, and bug fixes.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/new-harness`)
3. Submit a pull request with a description of the change

Please ensure new probes are clearly documented with expected behavior and the reasoning for inclusion.

---

## License

[Apache 2.0](LICENSE) — free to use, modify, and distribute with attribution.

---

## About

Safe AI for Humanity Foundation is an independent nonprofit research organization dedicated to understanding, measuring, and mitigating risks from advanced AI systems. All research is freely published.

- Website: [ai-4-h.org](https://ai-4-h.org)
- Email: [info@ai-4-h.org](mailto:info@ai-4-h.org)
- EIN: 41-4767005
