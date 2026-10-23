# Evaluation results

LLM: `gemini` / `gemini-3.5-flash` · embeddings: `BAAI/bge-small-en-v1.5` · top_k=5 · threshold=0.65

**5/5** questions gave the expected found / not-found result.

| # | Question | Expected | Top score | System output |
|---|----------|----------|-----------|---------------|
| 1 | What are the three main types of inspection? | answer ✅ | 0.726 | Answer: Maintenance inspection types may be broadly classified into standby, running, and disassembly inspections. Additionally, scheduled inspections for the GE Vernova Annular/Silo fleet are categorized as Type A Visual Inspection, Type B Visual Inspection, and Type C Major Inspection.<br>Source: Page 25 — Maintenance Inspections; Page 32 — ABC Inspections For Annular/Silo Fleet |
| 2 | What does a maintenance factor of two mean? | answer ✅ | 0.783 | Answer: A maintenance factor of two indicates a maintenance interval that is half of the baseline interval. Maintenance factors are established for operation that differs from the baseline to quantify the effect on component lives and provide the required increase in maintenance frequency.<br>Source: Page 9 — Major Factors Influencing Maintenance and Equipment Life |
| 3 | What is the purchase price of this gas turbine? | Not found ✅ | 0.682 | Not found |
| 4 | What is the melting point of titanium? | Not found ✅ | 0.576 | Not found |
| 5 | What is the recommended combustion-inspection interval in hours? | answer ✅ | 0.819 | Answer: The recommended combustion-inspection interval depends on the turbine model and combustor configuration. Baseline inspection intervals in Figure 36 range from 8,000 hours up to 32,000 hours (for example, 12,000 hours for 6B.03, 8,000 or 12,000 hours for 7E.03/9E.03, and 24,000 or 32,000 hours for various F-class DLN units). To determine the recommended inspection interval for a specific application, the baseline inspection interval is divided by the calculated maintenance factor.<br>Source: Page 36 — Combustion Inspection Interval; Page 33 — Inspection Intervals |
