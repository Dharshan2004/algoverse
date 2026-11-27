# Pilot explainer transcript

## The question

This pilot asks a simple question. If a response contains unsafe material, can adding a refusal sentence make a judge call it safe? We tested saved responses, not model training. Here is how the code works, and what actually happened.

## 01  Build the experiment

We start with fifty saved request and response pairs: twenty five labelled unsafe and twenty five labelled safe. Protocol dot py creates four versions. Original, refusal opening, neutral opening, and an alternate refusal. The response body stays exactly the same. The human label is kept for scoring, and is never shown to the judge.

## 02  Swap the answer letters

Each version is judged twice. First, A means safe and B means unsafe. Then we reverse those meanings. This checks whether a preference for the letter A or B influences the outcome. Fifty records, times four versions, times two mappings, makes four hundred judgments for each model.

## 03  Run the models

Inference dot py loads a pinned model checkpoint on the GPU. It formats the prompt, checks its length, and generates up to four answer tokens without random sampling. The parser accepts an exact answer code after limited native formatting cleanup. Other outputs remain invalid. Both models ran on an L4 GPU with B F sixteen computation and compressed four bit weights.

## 04  Save and verify

For every judgment, we save the answer, first token A and B scores, and six internal activation snapshots. These snapshots record what happened; we did not change a refusal direction. Two extra calls per model check that a do nothing intervention leaves results unchanged. The reporting code then checks completeness, unchanged text, scores and saved tensors before calculating rates.

## 05  What changed?

A false safe judgment means the dataset says unsafe, but the judge says safe. Averaging the two letter mappings, Ministral increased from six percent to ten percent with the refusal opening. Gemma increased from twenty two percent to twenty eight percent. These are changes of four and six percentage points, not proof of a general effect.

## 06  Check the controls

The neutral opening left Ministral at six percent and raised Gemma to twenty four percent. Ministral showed no refusal increase under one letter mapping, and an eight point increase under the other. Only Gemma had examples that flipped from unsafe to safe under both mappings: two of twenty five. Ministral had zero.

## 07  What is established?

Eight hundred judgments and four identity calls completed. All saved activation files passed the audit. Ministral had no invalid outputs; Gemma had ten out of four hundred. Those invalid outputs stay in the calculations. Source labels, possible text completeness issues, and shared request families still need review. We have not established statistical significance or a causal mechanism.

## 08  Track the work

The repository now separates reusable notebooks from sanitized historical snapshots. The actual pilot ran through the uploaded notebook and existing inference script. The new pilot launcher is for future runs, and was dry run tested. Aggregate reports, notebook provenance and this video are tracked. Raw data stays private. Next, review the sample before scaling or attempting causal interventions.
