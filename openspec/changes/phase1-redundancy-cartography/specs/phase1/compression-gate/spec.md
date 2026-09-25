# Spec Delta

## Purpose

Defines the Stage A quality and compression gate for post-hoc structural compression: the strict ±0.5 quality bar, how stored bytes are counted, the matched-quality comparison against integer quantization, the harness sanity check, the fixed healing budget, and one decision rule that is fixed before the run.

## ADDED Requirements

### Requirement: Strict quality bar

The Stage A gate MUST require, on the target model measured against its own BF16 reference, both a relative perplexity increase of at most 0.5 % and a delta of at least −0.5 points on every benchmark in the fixed evaluation set. Both conditions MUST hold simultaneously; a configuration failing either is not a passing configuration.

#### Scenario: Passing configuration

- **WHEN** a compressed configuration is scored against the BF16 reference
- **THEN** it passes the quality bar only if relative perplexity increase is ≤ 0.5 % and every benchmark delta is ≥ −0.5 points

#### Scenario: One condition fails

- **WHEN** relative perplexity increase is 0.3 % but one benchmark drops 0.8 points
- **THEN** the configuration fails the quality bar

### Requirement: Stored-byte accounting

Compression MUST be measured as stored bytes: the serialized size of every tensor needed to run the compressed model, including quantized weights, scales, zero points, codebooks, indices, sparse masks and any auxiliary arrays. Stored bytes MUST be normalised by the reference model's parameter count (bytes per parameter) and the compression ratio MUST be the ratio of the reference model's stored BF16 bytes to the compressed model's stored bytes. Bytes outside the compressed model, such as the evaluation harness or a calibration cache, MUST NOT be counted.

#### Scenario: Auxiliary tensors counted

- **WHEN** a configuration achieves a lower nominal weight bit-width by adding codebooks and indices
- **THEN** the codebooks and indices are included in its stored bytes before it is scored

#### Scenario: Ratio is defined against BF16

- **WHEN** the compression ratio of a configuration is reported
- **THEN** it equals reference stored bytes divided by compressed stored bytes for the same model on disk

### Requirement: Matched-quality integer baseline

Every Stage A claim MUST be compared, at equal or better quality, against integer group-128 quantization baselines at 4, 5 and 6 bits measured on the same model, slice and evaluation set. The comparison MUST report the smallest stored bytes among the baselines that satisfy the same quality bar, and the claimed method's stored bytes MUST be reported against that value.

#### Scenario: Baseline selected at matched quality

- **WHEN** two configurations have different quality
- **THEN** the baseline used for comparison is the smallest-byte integer configuration that meets the same quality bar, not the largest-byte one

#### Scenario: Comparison is not against FP16

- **WHEN** a compression claim is stated
- **THEN** its reference is the matched-quality integer baseline, and a comparison against FP16 alone is not a passing claim

### Requirement: Harness sanity check

An 8-bit group-128 quantized baseline MUST be evaluated before any verdict is issued. If its relative perplexity increase exceeds 0.1 %, the evaluation harness MUST be declared invalid, the run MUST be recorded as void, and no Stage A verdict may be produced from it.

#### Scenario: Sanity passes

- **WHEN** the 8-bit baseline is evaluated
- **THEN** a relative perplexity increase of at most 0.1 % validates the harness for that run

#### Scenario: Sanity fails

- **WHEN** the 8-bit baseline shows a relative perplexity increase above 0.1 %
- **THEN** the run is void and its numbers MUST NOT be used in a verdict

### Requirement: Fixed evaluation set

The evaluation set MUST be: perplexity on WikiText-2 (`wikitext-2-raw-v1`, test split, fixed slice and sequence length) plus the benchmarks ARC-Easy, HellaSwag, LAMBADA and GSM8K. The few-shot count per benchmark, the slice definition and the harness version MUST be recorded in the run's configuration before the run, and MUST NOT be changed between the reference measurement and the compressed measurement.

#### Scenario: Reference and compressed agree on the set

- **WHEN** BF16 reference and compressed model are evaluated
- **THEN** both use the identical slice, benchmark list, few-shot counts and harness version

#### Scenario: Half-precision reference

- **WHEN** a benchmark delta is reported
- **THEN** it is computed against the BF16 reference measured by the same harness configuration

### Requirement: Bounded healing

Recovery training MUST be optional, and if used it MUST be a single pre-registered run bounded by at most 1 GPU-hour, LoRA rank at most 8, and seeds taken from the configuration. Healing MUST be applied only after the removal mask is fixed, MUST NOT change which units were removed, and the gate result MUST be reported both before and after healing. The verdict MUST use the pre-healing measurement, and healing MUST NOT upgrade a configuration's verdict.

#### Scenario: Mask is frozen before healing

- **WHEN** a configuration is healed
- **THEN** the set of removed units is identical to the unhealed configuration and only adapter parameters were trained

#### Scenario: Both numbers reported

- **WHEN** a healed configuration is scored
- **THEN** the report contains the pre-healing and post-healing quality values and states which of them the verdict uses

### Requirement: Single Stage A decision rule

Exactly one GO / YELLOW / RED rule MUST be used, fixed before the run. GO when some configuration meets the quality bar with stored bytes at most 0.9 × the smallest matched-quality integer baseline's stored bytes and at least 2 × compression versus the BF16 reference. YELLOW when a configuration meets the quality bar with stored bytes between 0.9 × and 1.1 × of that baseline. RED otherwise, including the case where no configuration reaches 2 × compression at the quality bar. A configuration that meets the quality bar only after healing MUST be scored YELLOW at best, never GO. Altering a threshold MUST be recorded as a change to this spec, not applied after results are seen.

#### Scenario: Verdict is one of three

- **WHEN** Stage A results are scored
- **THEN** the verdict is exactly GO, YELLOW or RED, with RED as the default when no other condition holds

#### Scenario: Threshold change is recorded

- **WHEN** any threshold in this rule is altered
- **THEN** the alteration is recorded as a change to this spec before it is applied

### Requirement: Speed, VRAM and context excluded from Stage A

The Stage A gate MUST NOT include decode speed, peak VRAM or context length. These are acceptance criteria of later stages, and the Stage A report MUST state that they were not measured.

#### Scenario: Gate rests on quality and bytes only

- **WHEN** a Stage A verdict is produced
- **THEN** it rests on quality and stored bytes only, and the report states that speed, VRAM and context were not measured
