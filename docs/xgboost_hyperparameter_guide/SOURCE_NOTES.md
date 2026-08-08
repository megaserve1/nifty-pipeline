# XGBoost Guide — source and QA notes

## Reporting job

- Audience: technical.
- Question: explain how the current XGBoost classifier works and what every relevant parameter controls.
- Decision supported: code review, HPO design, and oral questioning.
- Scope: the current seven-class gbtree + hist pipeline, not regression, ranking, survival, DART, or linear-booster workflows.
- Success criterion: a first-time reader can distinguish tree-size, split-gating, leaf-regularization, sampling, learning-schedule, and runtime parameters without confusing internal loss with trading cost.

## Required technical-report structure mapping

1. Title: cover.
2. Technical summary: “Technical summary”.
3. Key findings/evidence: sections 1–16, using formulas, worked examples, and exact lookup tables.
4. Scope/definitions: sections 1–3 and the catalog scope note.
5. Methodology/model specification: sections 2–14.
6. Limitations/robustness: sections 17–18.
7. Recommended next steps: section 20.
8. Further questions: included under section 20.

## Evidence inventory

- Official XGBoost boosted-tree derivation:
  https://xgboost.readthedocs.io/en/stable/tutorials/model.html
- Official XGBoost parameter reference:
  https://xgboost.readthedocs.io/en/stable/parameter.html
- Official multiclass softmax objective example:
  https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html
- Official tree-method comparison:
  https://xgboost.readthedocs.io/en/stable/treemethod.html
- Official tuning notes:
  https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- Local configuration/code:
  configs/tuned/xgboost.json, configs/hyperparams.yaml,
  trainer/hyperparams.py, trainer/train.py, and trainer/hpo.py.

## Presentation choices

- No quantitative performance chart was included because no new training or model comparison was run.
- Formulas, worked split examples, process steps, and exact lookup tables are more faithful than a decorative chart for this conceptual/model-specification report.
- The report is self-contained and makes no network requests.

## Packaging note

The canonical Data Analytics portable-report packager could not run because this workspace has no Node.js or npm executable. The report therefore uses the documented static-HTML to local-Chrome PDF fallback. The HTML is the source of truth and must be retained beside the PDF.

## Verification result

- Rendered successfully with local Chrome headless: 17 A4 pages, 483,748 bytes, tagged and searchable.
- Extracted selectable text and confirmed all requested parameter names, formulas, headings, warnings, and official references.
- Rendered and visually inspected every page; no blank pages, clipped text, broken tables, or missing sections were found.
- Confirmed that no TODO, TBD, placeholder, or app-only interface text appears in the deliverable.
- `git diff --check` passed.
