# Certified Safety Curation for LLM Fine-Tuning

Code and archived results for *Clean Data, Unsafe Model: Certified Safety Curation for LLM
Fine-Tuning*.

A distribution-free certificate takes any scorer and a few hundred human labels and returns a score
cutoff whose kept set is at most an alpha-fraction harmful with probability 1 - delta, or refuses
when no cutoff qualifies. Because a certified set's harmful fraction is then known, the certificate
can be used to measure what else a filter did to the set; this repository holds the code and the
evaluation records behind that measurement.

Everything the paper reports regenerates from the files here. Nothing numeric is typed by hand.

<!-- AUTHORSHIP: this block is stripped from the anonymous archive built by
     scripts/make_anonymous_zip.sh. Keep author, contact and arXiv details inside it. -->
**Paper.** arXiv: _to be filled once the preprint is posted_.

**Authors.** Adam Haroon (aharoon@iastate.edu) and Cody Fleming, Iowa State University, Ames, IA, USA.

**Certificate.** The certificate applied here to text is from *Certified Safety Curation:
Distribution-Free Guarantees for Safe Offline Reinforcement Learning* (arXiv:2609.12014), whose code
is released separately.
<!-- END AUTHORSHIP -->

## Layout

    scripts/                 the experiment pipeline, one file per stage
      common.py              pool construction, the certificate, the scorers
      cert_rate_theory.py    exact closed-form certification probability
      step1_certify_gate.py  the certification gate on text
      step2_sft.py           0.5B full fine-tuning;  e3_lora_sft.py  8B LoRA
      step2_eval.py          generation and judging;  e6_utility.py  helpfulness, over-refusal
      saft_select.py         SAFT reimplementation;  seal_select.py  SEAL
      e*.py, run_e*.sh       the numbered experiments and their runners
    paper/scripts/           everything that turns results into the paper
      make_tables.py         regenerates every table
      make_figures.py        regenerates every figure;  concept_figure.py, qualitative.py
      completeness_check.py  the gate: experiment matrix plus every audit below
      audit_prose.py         every number in the prose, against the result files
      verify_constants.py    every number that is not a table cell, bound to its source
    results/                 per-condition evaluation records and per-generation judge verdicts
    selections/              each 0.5B selection as an index list into its pool, with its size,
                             harmful fraction, demonstration share, certification outcome, threshold
    selections_e3/           the same for the aligned-model setting, per contamination level
    paper/data/tables/       the generated tables
    paper/figures/           the generated figures

## Regenerating the paper's tables and figures

    export CDC_ROOT=$(pwd)                      # optional; otherwise inferred from the file's path
    python paper/scripts/make_tables.py             # 19 of the 21 tables
    python paper/scripts/judge_agreement.py         # the judge-agreement table
    python paper/scripts/label_complexity_bound.py  # the label-complexity table
    python paper/scripts/qualitative.py             # the two qualitative tables
    python paper/scripts/make_figures.py            # 9 of the 10 figures
    python paper/scripts/concept_figure.py          # Figure 1

A clean run reproduces the files already in `paper/data/tables` and `paper/figures` byte for byte,
except that figure PDFs carry a creation timestamp -- compare the PNGs. Figure rendering depends on
the matplotlib version; see `requirements.txt`.

Where a step would otherwise need something too large to distribute -- the study pool, the SAFT
embedding matrices, the raw generations -- the quantity it actually uses is cached in `results/`
and read from there, so the outputs are identical either way.

## Checking the numbers

    python paper/scripts/completeness_check.py

This is the gate. It checks that the experiment matrix is complete (every condition, every seed,
every judge) and runs `verify_constants.py`, which binds every number in the paper that is not a
table cell to the one expression that produces it. Two further rules read the paper's LaTeX source,
which is not part of this repository; they announce a skip and the rest of the gate still runs.

## Rerunning the experiments

The pipeline needs one 24 GB GPU. A 0.5B fine-tuning run takes one to three minutes, an 8B LoRA run
about fifteen, SEAL's selector fifty minutes at its published epoch count and nine hours
budget-matched. `scripts/run_*.sh` drive the stages; set `PY` (and `PY_LLM`, `PY_SEAL` where a
runner uses them) to the interpreters you want, or leave them to default to `python`.

## Data and models

Public throughout: BeaverTails and PKU-SafeRLHF for the pools, Alpaca and AlpaGasus for the quality
property, HEx-PHI, HarmBench, DirectHarm4 and XSTest for evaluation, and Qwen2.5, Llama-3.1,
TinyLlama, beaver-dam-7b, MD-Judge, WildGuard and Llama Guard 3 as the models and judges.
`scripts/common.py` fetches what it needs; HEx-PHI requires its authors' access form.

Not tracked here: fine-tuned model weights, raw generations and the datasets themselves (see
`.gitignore`). The per-generation judge verdicts the tables rest on are in `results/gen_scores/`,
and a 4000-example copy of each pool is in `results/`, so tables and figures regenerate without
downloading the datasets.
