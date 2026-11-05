# Frontier-model experiments (test split, 400 documents)

## Downstream quality (Δ vs raw, points, 95% paired-bootstrap CI)

| consumer | condition | N | QA | classify | extract | agreement err % |
|---|---|---|---|---|---|---|
| Opus 5.5, medium | raw (absolute %) | 400 | 100.0 | 94.5 | 98.8 | -- |
| Opus 5.5, medium | typed | 400 | -0.4 [-1.0, +0.0] | -3.8 [-6.2, -1.2] | -2.5 [-4.0, -1.0] | 2.6 [1.0, 4.3] |
| Opus 5.5, medium | opaque | 400 | +0.0 [+0.0, +0.0] | -3.0 [-5.5, -0.2] | -3.5 [-5.4, -1.7] | 4.9 [2.7, 7.1] |
| Opus 5.5, medium | direct_only | 400 | +0.0 [+0.0, +0.0] | -1.8 [-4.0, +0.5] | -1.7 [-3.1, -0.5] | 3.9 [2.1, 5.9] |
| Opus 5.5, medium | all | 400 | -0.1 [-0.4, +0.0] | -3.0 [-5.5, -0.5] | -2.6 [-4.1, -1.0] | 5.4 [3.2, 7.7] |
| Opus 5.5, medium | oracle | 400 | +0.0 [+0.0, +0.0] | +1.8 [+0.0, +3.8] | +0.0 [-0.4, +0.4] | 4.8 [2.9, 6.9] |
| Opus 5.5, low | raw (absolute %) | 400 | 100.0 | 92.8 | 98.7 | -- |
| Opus 5.5, low | typed | 400 | +0.0 [+0.0, +0.0] | -3.8 [-6.5, -1.0] | -2.5 [-4.1, -1.0] | 4.0 [2.1, 6.1] |
| Opus 5.5, low | opaque | 400 | -0.1 [-0.4, +0.0] | -4.0 [-7.0, -1.0] | -3.7 [-5.6, -1.8] | 4.9 [2.8, 7.0] |
| Opus 5.5, high | raw (absolute %) | 100 | 100.0 | 96.0 | 99.1 | -- |
| Opus 5.5, high | typed | 100 | +0.0 [+0.0, +0.0] | -4.0 [-9.0, +1.0] | -2.0 [-5.1, +0.0] | 3.3 [0.0, 7.1] |
| Sonnet 5.5, medium | raw (absolute %) | 400 | 100.0 | 87.0 | 98.7 | -- |
| Sonnet 5.5, medium | typed | 400 | -0.1 [-0.4, +0.0] | -2.8 [-5.8, +0.2] | -5.6 [-7.7, -3.5] | 3.4 [1.4, 5.7] |
| Sonnet 5.5, medium | opaque | 400 | -0.5 [-1.3, +0.0] | -3.8 [-7.0, -0.5] | -5.1 [-7.1, -3.1] | 4.4 [2.2, 6.8] |
| Haiku 5.5, medium | raw (absolute %) | 400 | 100.0 | 82.2 | 98.4 | -- |
| Haiku 5.5, medium | typed | 400 | -0.8 [-1.7, -0.1] | -2.0 [-5.0, +0.8] | -4.4 [-6.3, -2.4] | 5.6 [3.2, 8.5] |
| Haiku 5.5, medium | opaque | 400 | -1.3 [-2.4, -0.4] | -1.2 [-3.8, +1.2] | -5.4 [-7.6, -3.4] | 4.3 [2.2, 6.5] |

## Run-to-run variance of Δ(typed) (Opus 5.5 medium) and cross-judge

| run | N | QA Δ | classify Δ | extract Δ |
|---|---|---|---|---|
| primary, 400 | 400 | -0.4 [-1.0, +0.0] | -3.8 [-6.2, -1.2] | -2.5 [-4.0, -1.0] |
| rep3, 400 | 400 | -0.4 [-1.0, +0.0] | -3.0 [-5.8, -0.2] | -2.4 [-4.0, -1.0] |
| rep1, 100 | 100 | +0.0 [+0.0, +0.0] | -6.0 [-12.0, -1.0] | -2.0 [-5.1, +0.0] |
| rep2, 100 | 100 | +0.0 [+0.0, +0.0] | -5.0 [-10.0, +0.0] | -2.0 [-5.1, +0.0] |

- Opus-as-judge win rate vs raw, typed: 0.67 [0.63, 0.71] (n=400)
- Opus-as-judge win rate vs raw, opaque: 0.72 [0.68, 0.76] (n=400)

## Pairwise summary judge on the Opus 5.5 consumer run: win rate of the pseudonymized summary vs raw (0.5 = parity)

| judge | prompt | typed | opaque | oracle |
|---|---|---|---|---|
| Sonnet 5.5 | default | 0.68 [0.64, 0.72] | 0.73 [0.69, 0.76] | 0.75 [0.72, 0.79] |
| Opus 5.5 | default | 0.67 [0.63, 0.71] | 0.72 [0.68, 0.76] | -- |
| Qwen2.5-72B | default | 0.66 [0.63, 0.69] | 0.66 [0.64, 0.69] | -- |
| Qwen2.5-72B | completeness-blind | 0.39 [0.37, 0.41] | 0.40 [0.38, 0.43] | 0.48 [0.46, 0.51] |

## LLM-judged grammatical agreement (error rate among pseudonyms with gendered agreement)

| consumer run | condition | pseudonyms | with gendered agreement (%) | inconsistent | error rate % [CI] |
|---|---|---|---|---|---|
| haiku-5-5, medium | typed | 789 | 41 | 2 | 0.6 [0.0, 1.5] |
| haiku-5-5, medium | opaque | 747 | 22 | 3 | 1.9 [0.0, 4.4] |
| sonnet-5-5, medium | typed | 819 | 43 | 2 | 0.6 [0.0, 1.5] |
| sonnet-5-5, medium | opaque | 801 | 24 | 2 | 1.0 [0.0, 2.7] |

## Frontier model as its own anonymizer (llm_self) vs local systems — typed, direct+org+loc

| system | N | doc leak % [CI] | incl. partial | PER mention leak | over-redaction /1k | k̂<11 share |
|---|---|---|---|---|---|---|
| presidio | 400 | 64.2 [59.8, 68.8] | 71.8 [67.2, 76.0] | 17.5 [15.1, 20.1] | 15.9 | 13% |
| presidio+regex | 400 | 41.2 [36.5, 46.2] | 51.2 [46.5, 56.2] | 18.2 [15.7, 20.8] | 15.9 | 12% |
| presidio+casenorm+regex | 400 | 19.5 [15.5, 23.2] | 20.0 [16.0, 23.8] | 7.2 [5.5, 8.9] | 17.3 | 12% |
| wikineural+regex | 400 | 56.5 [51.2, 61.5] | 64.0 [59.0, 68.8] | 29.6 [26.5, 32.5] | 4.5 | 9% |
| wikineural+casenorm+regex | 400 | 11.0 [8.2, 14.2] | 11.5 [8.5, 14.8] | 2.8 [2.0, 3.7] | 5.1 | 8% |
| gliner+regex | 400 | 93.8 [91.5, 96.0] | 94.0 [91.8, 96.2] | 47.7 [45.6, 49.9] | 3.9 | 13% |
| finetuned+regex | 400 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 1.4 | 2% |
| finetuned-xt+regex | 400 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 31.0 | 2% |
| llm_self-haiku-5-5-low | 400 | 0.2 [0.0, 0.8] | 0.5 [0.0, 1.2] | 0.0 [0.0, 0.0] | 4.7 | 2% |
| llm_self-opus-5-5-low | 400 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 0.3 | 2% |
| llm_self-opus-5-5-medium | 400 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 1.6 | 2% |
| llm_self-sonnet-5-5-low | 400 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 4.2 | 2% |
| oracle | 400 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 0.0 | 2% |

## LLM attacker (Opus 5.5): attribute recall % on the data subject, surname recovery, inference-based k̂

| condition | gender | age | city | profession | employer | nationality | marital_status | diagnosis | surname | k̂ median | k̂<11 | k̂<100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw text (ceiling) | 100 | 100 | 88 | 66 | 90 | 90 | 26 | 100 | 100.0 | 1 | 64% | 77% |
| typed d+o+l | 100 | 69 | 2 | 66 | 36 | 35 | 25 | 100 | 35.0 | 8,295 | 14% | 26% |
| opaque d+o+l | 100 | 69 | 3 | 66 | 36 | 32 | 25 | 100 | 36.0 | 7,937 | 14% | 26% |
| typed direct\_only | 100 | 69 | 88 | 66 | 86 | 74 | 25 | 100 | 47.2 | 18 | 46% | 72% |
| typed all | 100 | 68 | 3 | 66 | 36 | 34 | 25 | 100 | 35.0 | 8,769 | 14% | 25% |
| oracle typed d+o+l | 100 | 68 | 0 | 66 | 0 | 10 | 26 | 100 | 0.0 | 47,287 | 2% | 6% |
| typed d+o+l, Sonnet attacker | 100 | 69 | 2 | 66 | 36 | 26 | 26 | 100 | 34.8 | 8,399 | 14% | 26% |
| typed d+o+l, Qwen2.5-14B attacker (local) | 98 | 64 | 2 | 64 | 24 | 26 | 26 | 99 | 31.5 | 19,429 | 11% | 19% |
| raw text, Qwen2.5-14B attacker (100 docs) | 100 | 98 | 79 | 67 | 92 | 47 | 31 | 100 | 100.0 | 1 | 65% | 82% |
| oracle, Qwen2.5-14B attacker (100 docs) | 97 | 68 | 0 | 67 | 0 | 0 | 30 | 96 | 0.0 | 33,380 | 1% | 8% |
| typed d+o+l, Qwen2.5-72B attacker (RunPod) | 98 | 68 | 2 | 66 | 26 | 26 | 26 | 99 | 33.8 | 15,342 | 12% | 20% |
| opaque d+o+l, Qwen2.5-72B attacker | 93 | 68 | 3 | 67 | 35 | 26 | 26 | 100 | 34.8 | 9,531 | 14% | 25% |
| typed direct\_only, Qwen2.5-72B attacker | 100 | 67 | 83 | 65 | 84 | 26 | 26 | 100 | 45.0 | 20 | 44% | 71% |
| raw text, Qwen2.5-72B attacker | 100 | 100 | 84 | 67 | 90 | 26 | 28 | 100 | 100.0 | 1 | 64% | 77% |
| oracle, Qwen2.5-72B attacker | 100 | 69 | 0 | 66 | 0 | 0 | 28 | 100 | 0.0 | 45,929 | 2% | 6% |

### Surface survival vs attacker recall (typed d+o+l, Presidio + regex; both among documents where the attribute applies)

| attribute | surface survival % | attacker recall % | attacker precision % |
|---|---|---|---|
| gender | 100 | 100 | 100 |
| age | 69 | 69 | 100 |
| city | 5 | 2 | 8 |
| profession | 67 | 66 | 98 |
| employer | 30 | 36 | 100 |
| nationality | 26 | 35 | 99 |
| marital_status | 21 | 25 | 93 |
| diagnosis | 100 | 100 | 100 |

## Spend (list prices, from the call cache)

```
model                    calls      in_tok     out_tok      USD
claude-haiku-5-5          6400   4,125,486   2,055,275     1.44
claude-opus-5-5          31708  21,786,432   6,269,875   212.54
claude-sonnet-5-5        19110  19,039,380   2,912,684    67.21
ollama:qwen2.5:14b         600     450,708      53,049     0.00
openai:qwen72b            6395   6,117,124     231,041     0.00
TOTAL                    64213                           281.19
```

