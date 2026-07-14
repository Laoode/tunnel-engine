# Guardrail Benchmark — XGuard 0.6B

Model: `Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B` · threshold 0.5 · concurrency 8 · 357 samples
(dataset: DeepSeek v4 Pro generated, Sonnet 5 judged; Indonesian).

## Latency (guard check, the per-request hot-path cost)
- mean 102.4 ms | p50 98.7 | p90 116.3 | p99 271.3 | max 273.1

## Accuracy (blocked = predicted unsafe)
| Metric | Value |
|--------|-------|
| Accuracy | 83.2% |
| Precision | 99.6% |
| Recall (unsafe caught) | 82.3% |
| F1 | 0.901 |
| False-positive rate (safe blocked) | 4.2% |

Confusion: TP 274 · FP 1 · TN 23 · FN 59

## Per-category recall (unsafe)
| Category | Caught | Recall |
|----------|--------|--------|
| Hate Speech-Cyberbullying | 7/12 | 58% |
| Ethics and Morality-Social Ethics | 7/12 | 58% |
| Inappropriate Suggestions-Medicine | 7/12 | 58% |
| Hate Speech-Threats and Intimidation | 7/11 | 64% |
| Extremism-Social Disruption | 8/12 | 67% |
| Risks Involving Minors-Minor Abuse and Exploitation | 8/12 | 67% |
| Crimes and Illegal Activities-Property Infringement | 9/12 | 75% |
| Physical and Mental Health-Mental Health | 9/12 | 75% |
| Data Privacy-Commercial Secret | 9/12 | 75% |
| Cybersecurity-Physical Security | 9/12 | 75% |
| Extremism-Extremist Ideological Trends | 9/12 | 75% |
| Inappropriate Suggestions-Law | 9/12 | 75% |
| Hate Speech-Abusive Curses | 9/11 | 82% |
| Ethics and Morality-Science Ethics | 10/12 | 83% |
| Cybersecurity-Access Control | 10/12 | 83% |
| Risks Involving Minors-Minor Delinquency | 10/12 | 83% |
| Crimes and Illegal Activities-Pornographic Contraband | 11/12 | 92% |
| Crimes and Illegal Activities-Dangerous Weapons | 11/12 | 92% |
| Hate Speech-Defamation | 11/12 | 92% |
| Physical and Mental Health-Physical Health | 11/12 | 92% |
| Extremism-Violent Terrorist Activities | 11/12 | 92% |
| Risks Involving Minors-Corruption of Minors | 11/12 | 92% |
| Crimes and Illegal Activities-Drug Crimes | 12/12 | 100% |
| Crimes and Illegal Activities-Economic Crimes | 12/12 | 100% |
| Data Privacy-Personal Privacy | 11/11 | 100% |
| Cybersecurity-Malicious Code | 12/12 | 100% |
| Cybersecurity-Hacker Attack | 12/12 | 100% |
| Inappropriate Suggestions-Finance | 12/12 | 100% |
