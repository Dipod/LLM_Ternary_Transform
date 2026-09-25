# Tasks

## 1. Setup and fact-closing

- [x] 1.1 Create the `phase1_compression/` package (`cartography/`, `transform/`, `evaluation/`, `main.py`, `config.yaml`) and `tests/phase1_compression/`; verify `ruff check .`, `ruff format --check .`, `mypy .` and `pytest` pass on the skeleton with an import smoke test.
- [x] 1.2 Add the seed helper that reads all seeds from `config.yaml` (torch, numpy, random, deterministic-algorithms flag); verify a unit test that two runs under the same config produce bit-identical tensors and that no algorithm calls a seed function itself.
- [x] 1.3 Read `Qwen/Qwen3-8B`'s `config.json` and record `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `hidden_size` into the run provenance; verify the recorded values equal the file's values and the config loads through `AutoConfig` without fetching weights.
- [x] 1.4 Download `Qwen/Qwen3-8B` into the HF cache and verify the on-disk footprint is within the stated disk budget and a forward pass runs at `seq_len = 512` on the ROCm device; record device, dtype and peak VRAM.
- [x] 1.5 Close the harness fact: run `lm_eval` on the cached 0.5B with `--tasks arc_easy --limit 5`, verify it prints an accuracy and record the installed version; if it cannot be installed against our pins, record the blocker and stop before any gate task.

## 2. Evaluation harness, accounting and baselines

- [x] 2.1 Reuse the Phase 0 slice builder for the fixed WikiText-2 slice and the benign KL slice; verify the slice reports 65,536 non-pad tokens and that the fit/held-out split is reproducible across two invocations.
- [x] 2.2 Implement the relative-perplexity metric on the fixed slice; verify with a known-answer test (an unmodified copy scores ΔPPL = 0 within tolerance, a deliberately degraded copy scores above the bar) and state the tolerance.
- [x] 2.3 Implement stored-byte accounting over every tensor needed to run a model (weights, scales, zero points, codebooks, indices, masks); verify a unit test where a lower nominal bit-width with large codebooks does not score lower than a simple 4-bit configuration.
- [x] 2.4 Implement the matched-quality integer baseline selection (smallest stored bytes among group-128 4/5/6-bit configurations meeting the quality bar); verify a unit test on synthetic (bytes, quality) pairs covering the boundary where a larger-byte configuration has better quality.
- [ ] 2.5 Measure the 8-bit group-128 sanity baseline and the 4/5/6-bit baselines on the target model; verify the 8-bit baseline's relative perplexity increase is ≤ 0.1 %, record the harness as valid for the stage, and record every baseline run in the three-file layout.
- [ ] 2.6 Evaluate the BF16 reference and every baseline on ARC-Easy, HellaSwag, LAMBADA and GSM8K; verify all are run with the identical task list, few-shot counts and harness version recorded in the config, and that each delta is computed against the same BF16 reference.

## 3. Redundancy cartography on the small model

- [x] 3.1 Implement structured-component removal (blocks, attention heads, FFN intermediate dimensions) with iterative greedy importance scoring; verify the record contains the unit ranking, the removal order and the first step that exceeds the proxy budget, plus the `E_x` cross-check at that step.
- [x] 3.2 Implement the direction/subspace axis (difference-in-means + PCA/SOM basis, KL score on the benign slice, grid over direction, layer band and strength); verify the record lists every evaluated candidate with its KL value and that removal is applied as a weight orthogonalization, not an inference hook.
- [x] 3.3 Implement the ternary axis on a TCD decomposition (`W ≈ B diag(D) C`): plane dropping, N:M masking over the natural zeros, ternary-core rank reduction; verify the observed zero fraction is recorded first, two or more N:M patterns are compared, and any re-fit uses the decomposition's closed-form with no gradient step.
- [x] 3.4 Run the three axes under the finite budget (≤ 4 GPU-hours each, ≤ 1 working day total) and write the map; verify `results/<map-id>/{config.yaml,metrics.json,report.md}` exist with the per-axis curves and that each axis records budget-stop as its stopping reason when it hits the cap.

## 4. Combined probe on the target model

- [x] 4.1 Recompute the unit selection on `Qwen/Qwen3-8B` with the same procedure and freeze the ordered transformation list into the probe config; verify the probe record references the map id and contains the recomputed selection, not a copy of the small-model unit list.
- [ ] 4.2 Run the combined probe pre-healing in the frozen order; verify the record contains quality (perplexity and all four benchmarks), stored bytes and bytes-per-parameter, and the comparison against the matched-quality integer baseline.
- [x] 4.3 Implement and unit-test the single decision rule on synthetic boundary cases (just inside/outside 0.9 × and 1.1 ×, and below 2 × compression); verify the tests fail against a rule with any threshold altered.
- [ ] 4.4 Run the bounded mini-LoRA healing (≤ 1 GPU-hour, rank ≤ 8, seeds from config) with the removal mask frozen; verify the removed-unit set is identical to 4.2, no unit selection changed, and both pre- and post-healing quality values are recorded with the verdict attributed to the pre-healing value.
- [ ] 4.5 Score the Stage A verdict under the single rule and record it; verify the record states GO/YELLOW/RED, the thresholds used, the matched-quality baseline bytes, and that a configuration passing only after healing is scored no higher than YELLOW.

## 5. Integration and closure

- [x] 5.1 Write the Stage A verdict into `results/decision_report.md`; verify the entry names the verdict, the map and probe experiment ids, and states that decode speed, peak VRAM and context length were not measured.
- [x] 5.2 Update `openspec/project.md` (Phase 0 closed, Phase 1 Stage A open with its verdict) and `memory.md`; verify the wording matches the recorded verdict and that no Phase 0 or STOP record was reinterpreted.
- [x] 5.3 Run the full gate chain on the final content and verify `ruff check .` clean, `ruff format --check .` clean, `mypy .` clean and `pytest` green with counts recorded in the delivery report; confirm no gate was skipped and no test was deselected without a stated reason.
