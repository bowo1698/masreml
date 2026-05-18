# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package does

`masreml` is an R package implementing REML-BLUP genomic prediction for both bi-allelic SNP and multi-allelic microhaplotype (MH) markers. It supports four relationship matrix types (SNP additive, SNP dominance, MH additive, pedigree A), multiple REML algorithms (AI, EM, HE, HI), binary traits via Laplace approximation, EMMAX-based GWAS, and GWAS-assisted prediction (GWABLUP). All numerical computation is handled by a Rust backend via `extendr`.

**S3 methods on the returned `masreml` fit object:**
- `summary(fit)` — full report including a **Training Performance** block (R2, RMSE, accuracy or AUC, calibration slope as `bias`).
- `predict(fit, ..., y_new = ...)` — GEBVs + auto-computed test-set metrics in one call (mirrors `masbayes::predict()`).
- `print(fit)` — brief one-screen summary.

`masreml()` also prints an automatic post-fit summary banner on completion (gated by `verbose = TRUE`), parallel to `masbayes::run_bayesr()` / `run_bayesa()`.

## Commit policy (read first)

**All commits to this repository are authored by the user manually. Claude never runs `git commit` or `git push` here.** When asked to draft a commit message, prepare a release note, or generate any text that the user will paste into Git:

- **Do not** append `Co-Authored-By: Claude <noreply@anthropic.com>` (or any variant). The repo must not carry Anthropic / Claude co-author trailers in its history.
- **Do not** append the `🤖 Generated with [Claude Code](https://claude.com/claude-code)` footer (or any variant of it).
- **Do not** mention Claude, Anthropic, AI assistance, or the assistant in the commit message body, subject line, or release notes that will be pushed.
- Output every suggested commit message as if it were written solely by the user.

This also applies to PR descriptions, tag annotations (`git tag -a`), and any RELEASE_NOTES_v*.txt that will be shipped with the package — those files are pushed to GitHub and must not advertise AI authorship.

Additional Git hygiene specific to this repo:
- `CLAUDE.md`, `PLAN.md`, `REVISION_PLAN.md`, and `*.bak` files are local-only. Never `git add` them, even when they appear in `git status`. The user manages `.gitignore` manually; do not edit it.
- The `docs/` tree under the parent project (planning notes, design specs) is also local-only and must never be staged from inside this package.

## Requirements

- **Rust** (≥ 1.70.0): Install via `rustup`. Required to compile the Rust backend.
- **R** (≥ 4.2)
- **OpenBLAS**: Required by `ndarray-linalg` for linear algebra. The `configure` script detects this.
- R packages: `devtools`, `rextendr`, `MASS`

## Build and development commands

```r
# After editing Rust code: recompile Rust and regenerate R wrappers
rextendr::document()

# After editing R code only: reload without recompiling Rust
devtools::load_all()

# Generate documentation from roxygen2 comments
devtools::document()

# Run tests
devtools::test()

# Install locally
devtools::install()

# Install from GitHub
devtools::install_github("bowo1698/masreml")
```

**Important**: Any change to Rust source files (`src/rust/src/**`) requires `rextendr::document()` to recompile. Changes to R files only need `devtools::load_all()`.

## GitHub Actions release workflow

`.github/workflows/build-release.yml` builds and publishes pre-compiled binaries for four platforms.

**Trigger:** `push: tags: 'v*'` or manual `workflow_dispatch` (the latter used for dry runs against a non-tagged branch).

**Build matrix** (`fail-fast: false`):
| runner | rust-target | ext | arch |
|---|---|---|---|
| `windows-latest` | `x86_64-pc-windows-gnu` | zip | x64 |
| `macos-13` (Intel) | `x86_64-apple-darwin` | tar.gz | x64 |
| `macos-14` (Apple Silicon) | `aarch64-apple-darwin` | tar.gz | arm64 |
| `ubuntu-latest` | `x86_64-unknown-linux-gnu` | tar.gz | x64 |

**Deliberate divergence from masbayes' workflow:** uses `macos-13` for the Intel target (not `macos-latest`). `macos-latest` is now Apple Silicon, and Homebrew's `openblas` is arch-specific — building x86_64 on an arm64 runner would link the wrong BLAS arch. macOS runners are therefore pinned to the matching architecture.

**Per-OS system deps:**
- **Linux:** `apt-get install libcurl4-openssl-dev libopenblas-dev gfortran pkg-config`.
- **macOS:** `brew install openblas`, then `PKG_CONFIG_PATH` / `LDFLAGS` / `CPPFLAGS` exported via `$GITHUB_ENV` so `ndarray-linalg`'s `openblas-system` feature finds the keg-only install.
- **Windows:** `r-lib/actions/setup-r@v2` with `rtools-version: '43'` supplies the mingw toolchain (gcc, gfortran, perl, make); `openblas-static` builds OpenBLAS from source — slowest matrix entry (~10–15 min).

**Per-OS pipeline:** checkout → setup-R (split macOS/Linux vs Windows+Rtools) → setup-Rust (with `targets: ${{ matrix.config.rust-target }}`) → install sys-deps → `remotes::install_deps()` → `rextendr::rust_sitrep()` (diagnostic) → Windows-only Rust target clean → `R CMD INSTALL . --library=tmp_lib` → strip `tmp_lib/masreml/src/` + `Makevars*` → tar.gz or zip → **smoke test** → rename → upload artefact.

**Smoke test step** (added on top of the masbayes workflow pattern; not present in masbayes): loads the freshly built binary from `tmp_lib`, calls `load_data()`, fits a small AI-REML model on the bundled training split — `y = d$pheno$y_cont_qtl_snp[d$train_idx]`, `markers = list(snp_add = d$snp[d$train_idx, ])`, `method = "AI"`, `verbose = FALSE` — and asserts `isTRUE(fit$converged)`, `is.finite(fit$varcomp$h2[["snp_add"]])`, `length(fit$gebv$snp_add) == n_train`, and `predict(fit)$GEBV` length. Failure aborts the matrix entry → no artefact uploaded → release job skips. Catches silent BLAS-linking failures that build cleanly but crash at runtime.

**Release job** (`needs: build`, `if: startsWith(github.ref, 'refs/tags/') || github.event_name == 'workflow_dispatch'`): downloads all artefacts via `pattern: pkg-*`, `merge-multiple: true`, then publishes via `softprops/action-gh-release@v2` with `tag_name: ${{ github.ref_name }}`, `permissions: contents: write`. On `workflow_dispatch` against a non-tag branch the release is tagged with the branch name (delete from Releases UI after dry run).

**Artefact naming:** `masreml_<arch>_<rust-target>.<ext>` — matches the masbayes convention.

**Manual release flow:**
1. `git push origin <branch>`, trigger `workflow_dispatch` from the Actions UI → dry run. Iterate fixes if any matrix entry fails.
2. Merge to `main`.
3. `git tag -a vX.Y.Z -m "Release vX.Y.Z"` matching `DESCRIPTION` version, then `git push origin vX.Y.Z` → triggers the real release.

**Design + plan documents** (local-only, not in the masreml repo): `../docs/superpowers/specs/2026-05-16-masreml-build-release-workflow-design.md` and `../docs/superpowers/plans/2026-05-16-masreml-build-release-workflow.md`.

## Architecture

The package has a strict two-layer design:

### R layer (public API + orchestration)
- **`R/masreml.R`** — Main entry point: `masreml()`. Validates inputs, builds G matrices via `.build_g_matrices()`, dispatches binary vs continuous trait, calls `.run_reml()` then `.solve_blup()`, wraps timing in `system.time()`, attaches `runtime` + `training_metrics` + `rds_path`, and prints auto-summary banner if `verbose = TRUE`.
- **`R/build_matrix.R`** — Routes marker inputs (raw `markers` list or pre-built `G` list) to Rust builders. Handles SNP (`.build_g_snp()`), MH (`.build_g_mh()`), and validation. Implements MH haplotype parsing (`.parse_mh_chr()`) and allele re-encoding for train/test splits.
- **`R/reml.R`** — Thin wrapper: converts R data to Rust-expected format, calls `r_run_reml()`, parses result.
- **`R/blup.R`** — Thin wrapper: calls `r_solve_ebv()` for EBV computation.
- **`R/predict.R`** — `predict.masreml()` (three modes: in-sample / pre-built G_full / raw markers_new), `evaluate_prediction()`, and `print.masreml_pred()`. Predict output is aligned with `masbayes::predict.masbayes_*()` for cross-package workflows.
- **`R/crossval.R`** — `cv_masreml()`: k-fold or LOO cross-validation. Rebuilds G once then subsets per fold. Calls `.run_reml()`/`.solve_blup()` directly (does NOT call `masreml()`), so the auto-summary banner does not fire per fold.
- **`R/gwas.R`** — `run_gwas()`: EMMAX GWAS for SNP or MH markers, plus `r_smooth_and_pp()` for posterior probabilities.
- **`R/gwablup.R`** — `gwablup()`: GWAS-assisted GBLUP via weighted G matrix. Calls `masreml()` internally with `verbose = FALSE` to avoid double-banner output.
- **`R/utils.R`** — Public API helpers: `build_G_snp()`, `build_G_mh()`, `build_D_snp()`, `build_A_ped()`, `varcomp()`, `compute_accuracy()`, link functions for binary trait.
- **`R/binary.R`** — Binary trait GLMM via single-step Laplace approximation (`.masreml_binary()`). Wraps timing in `system.time()`, computes observed-scale `training_metrics` from `(y, mu_hat)`, attaches `runtime` + `rds_path`, and calls `.print_run_summary()`.
- **`R/summary.R`** — S3 `print`/`summary` methods for `masreml`, `masreml_cv`, `masreml_pred`, plus internal helpers:
  - `.print_run_summary(x)` — auto post-fit banner (called by `masreml()` and `.masreml_binary()`).
  - `.maybe_save_rds_masreml(fit, save_rds, save_path)` — opt-in RDS persistence.
  - `.compute_training_metrics(y, y_hat)` — Gaussian-style metrics list (R2, RMSE, accuracy, bias). Used by both continuous and binary paths; binary feeds `(y_observed, mu_hat)` so `bias` is the calibration slope.
- **`R/zzz-rust-wrappers.R`** — `.Call` wrappers: one function per Rust entry point. Names match `wrap__r_*` symbols generated by `extendr`.
- **`R/zzz.R`** — Package load hook for `extendr`.

### Rust layer (`src/rust/src/`)
- **`lib.rs`** — All `#[extendr]` entry points exposed to R. Handles R↔Rust data conversion (RMatrix, ndarray).
- **`matrix/`** — G matrix builders: `snp_additive.rs` (VanRaden 2008), `snp_dominance.rs` (Da et al. 2015), `mh_additive.rs` (Da 2015 multi-allelic W_ah coding), `pedigree.rs` (Henderson 1976 tabular method).
- **`reml/`** — REML algorithms: `he_regression.rs`, `ai_reml.rs`, `em_reml.rs`, `adaptive.rs` (selects/switches algorithms; shares Cholesky factorization across gradient, BLUP, and GWAS).
- **`solver/`** — EBV solvers: `cholesky.rs` (n < 10,000), `pcg.rs` (n ≥ 10,000 preconditioned conjugate gradient), `factorized.rs` (`FactorizedV` struct reuses V factorization across REML and GWAS).
- **`gwas/`** — `emmax.rs` (EMMAX association test for SNP and MH), `smoother.rs` (moving average + posterior probability).
- **`utils/linalg.rs`** — Shared linear algebra primitives.

### Key design invariants
- The single Cholesky factorization of **V** is shared across REML gradient computation, BLUP solving, and EMMAX GWAS — avoiding redundant O(n³) operations.
- Named R vectors are the primary ID alignment mechanism. Always pass named `y` vectors; marker rownames must match `y` names.
- Two marker input modes exist: (1) raw marker data → Rust builds G internally, or (2) pre-built G matrices passed directly. Both routes converge at the same solver.
- For train/test splits, `ref_W` / `ref_mh` arguments ensure allele frequencies are computed from training only, preventing data leakage.
- The `markers` list argument accepts named elements: `snp_add`, `snp_dom`, `mh_add`. The `G` list accepts `snp_add`, `snp_dom`, `mh_add`, `pedigree`.
- **Binary metrics scale convention**: All binary `training_metrics` and `predict()$metrics` are computed on the **observed (probability) scale** — `cor(y_01, mu_hat)`, `lm(y_01 ~ mu_hat)`. This makes `bias` the **calibration slope** (1.0 = ideal, <1 over-dispersion, >1 under-dispersion). AUC is unaffected (rank-invariant).

## Key `masreml()` parameters

| Parameter | Default | Notes |
|---|---|---|
| `y` | — | Numeric phenotype vector (named). Binary: 0/1 only. |
| `X` | `NULL` | Fixed-effects design (n × c). NULL → intercept only. Do not include intercept column when supplying. |
| `markers` | `NULL` | Raw marker list: `snp_add`, `snp_dom`, `mh_add`. |
| `G` | `NULL` | Pre-built G list: `snp_add`, `snp_dom`, `mh_add`, `pedigree`. |
| `method` | `"auto"` | `"AI"`/`"EM"`/`"HE"`/`"HI"`/`"auto"`. Continuous → AI; binary → HE. |
| `solver` | `"auto"` | `"cholesky"` (n < 10000) or `"pcg"` (otherwise). |
| `trait` | `"continuous"` | `"binary"` triggers Laplace 1-step on liability. |
| `link` | `"logit"` | Binary only: `"logit"` or `"probit"`. |
| `verbose` | `TRUE` | Print auto post-fit summary banner. Set FALSE inside CV/loops. |
| `save_rds` | `FALSE` | Auto-save fit to disk. **Differs from masbayes** — default OFF because masreml is commonly called inside CV loops. |
| `save_path` | `NULL` | Override default `"results_masreml.Rds"` path. Ignored unless `save_rds = TRUE`. |
| `max_iter` | `100L` | REML iterations. |
| `tol` | `1e-6` | REML convergence tolerance. |
| `n_threads` | `NULL` | Defaults to physical cores. |

## Fit object structure (returned by `masreml()`)

| Field | Description |
|---|---|
| `gebv` | Named list of per-component GEBV vectors |
| `total_gebv` | Total GEBV (sum across components). Liability scale for binary. |
| `fixed_effects` | Fixed-effect coefficients |
| `varcomp` | List `sigma2` (per-component + residual) and `h2` (per-component) |
| `loglik` | Restricted log-likelihood |
| `algorithm` | REML algorithm string. For binary: `"Laplace-1step (HE)"` etc. |
| `solver` | Solver used (`"cholesky"` or `"pcg"`) |
| `converged`, `n_iter` | Convergence flag and REML iterations |
| `n` | Number of individuals |
| `runtime` | Elapsed seconds (REML + BLUP) |
| `training_metrics` | List: `R2`, `RMSE`, `accuracy`, `bias` (continuous) or `R2, RMSE, accuracy, bias, AUC, scale` (binary, observed-scale) |
| `rds_path` | Saved RDS path or `NULL` |
| `binary` (binary only) | Sub-list: `link`, `prevalence`, `h2_liability`, `h2_observed`, `auc`, `fitted` (probabilities), `n_iter`, `converged` |
| `call` | Matched call |
| `class` | `"masreml"` |

## `predict.masreml()` output structure

Aligned with `masbayes::predict.masbayes_*()` (primary names) plus backward-compat aliases.

| Field | Description |
|---|---|
| `GEBV` | Total GEBV vector. Primary name (alias: `total_gebv`). Liability scale for binary. |
| `total_gebv` | Alias for `GEBV` (back-compat) |
| `gebv` | Per-component GEBV list (masreml-specific) |
| `prob` | Predicted P(y=1) for binary. Primary name (alias: `fitted`). NULL for continuous. |
| `fitted` | Alias for `prob` (back-compat) |
| `metrics` | Auto-computed list if `y_new` supplied: `R2`, `RMSE`, `accuracy`/`AUC`, `bias` (calibration slope for binary), `r_MG`. Else `NULL`. |
| `h2`, `sigma2` | Carried from fit |
| `response_type` | `"continuous"` or `"binary"` |
| `algorithm` | REML algorithm from fit |
| `eval_scope` | `"in-sample (training)"`, `"test set (G_full)"`, or `"test set (markers_new)"` |
| `has_truth` | `TRUE` if metrics were computed |
| `train_ids`, `test_ids`, `n_train`, `n_test` | ID vectors and counts |
| `class` | `"masreml_pred"` |

**Three predict modes:**
1. **In-sample** — `predict(fit)` with no test-data args → returns training info plus `fit$training_metrics`. Mirror of `masbayes::predict(fit)` shortcut.
2. **G_full** — `predict(fit, G_full = ..., train_ids = ..., test_ids = ...)` for pre-built relationship matrices.
3. **markers_new** — `predict(fit, markers_new = ..., markers_train = ..., y_new = ...)` for raw marker test data; allele frequencies derived from `markers_train` to prevent leakage.

`y_new` (optional) triggers auto-metric computation via `evaluate_prediction()` (still exported as a standalone helper for arbitrary GEBV vectors).

## `evaluate_prediction()` helper

Standalone metrics computation for any GEBV vector. Signature: `evaluate_prediction(gebv, y, h2 = NULL, tbv = NULL, fitted_prob = NULL)`. Returns a single-row data.frame with `r_test_y`, `r_test_g`, `bias`, `r_MG`, `AUC`, `RMSE`.

**Binary scale convention**: when `fitted_prob` is supplied, `r_test_y`, `bias`, and `RMSE` are computed on the observed (probability) scale → `bias` is the calibration slope. `r_test_g` and `r_MG` remain on the GEBV (liability) scale because TBV and h² are defined at the genetic-value level. AUC always uses `fitted_prob` (rank-invariant).

## Auto post-fit summary banner

`masreml()` prints a banner on completion (when `verbose = TRUE`) summarising:
- Algorithm + trait tag
- Observations, components, runtime, solver, RDS path (if saved)
- REML convergence: TRUE/FALSE + iterations + log-likelihood
- Variance components table with σ², %var, h²
- Total h² (genetic for continuous; liability + observed for binary)
- Binary trait block (prevalence, AUC, fitted P range) — binary only
- GEBV summary line

Implementation: `.print_run_summary()` in `R/summary.R`. Continuous and binary branches share the skeleton; binary adds the trait block and uses different total-h² label.

## Bundled demo dataset (`load_data()`)

`load_data()` returns a deterministic family-structured dataset bundled at `inst/extdata/demo_data.rds`: **n=200 measured individuals (10 full-sib families × 20 offspring) + 20 founder parents in pedigree only**, 500 SNPs in 250 blocks of 2 SNPs each, 20 QTL with `rnorm` effects, h² target 0.5 (realised SNP h² ≈ 0.52, MH h² ≈ 0.47). Includes `snp` (dosage), `mh` (haplotype matrix consumable directly by `build_G_mh()` via auto-detect), `pheno` with 4 trait variants + TBVs + balanced `sex` fixed effect, a 220-row `pedigree` (use with `build_A_ped()`), and **physical-position maps** `map_snp` (columns: `SNP`, `CHROM`, `POS`) / `map_mh` (columns: `block_id`, `chr`, `start_pos`, `end_pos`, `n_snps`; schema mirrors maspipeline's `microhaplotype_coordinates.csv`). The maps are a synthetic 5-chromosome layout with 100 kb intra-chr spacing — deterministic, no RNG. A **within-family 80/20 train/test split is bundled** as `d$train_idx` (length 160, 16 per family) and `d$test_idx` (length 40, 4 per family). Demo accuracy: continuous `r_test_g` ≈ 0.75 (GBLUP), binary AUC ≈ 0.80. The same Rds is shipped (byte-identical) with `masbayes`. Generator at `genomic_prediction/tools/make_demo_data.R`; regenerate with `Rscript tools/make_demo_data.R` and reinstall both packages.

Within-family prediction accuracy reflects the breeding-scenario use case (predicting siblings of training animals), NOT the harder problem of generalising to unrelated populations. For aggregate / more-stable accuracy estimates, `cv_masreml()` (continuous only) provides k-fold or LOO cross-validation using all n=200 observations.

## Examples

The `examples/` directory contains complete worked examples:
- **`01_basic_continuous_trait.R`** — REML-BLUP for continuous trait with SNP markers (Mode A: raw markers, Mode B: pre-built G). Uses `load_data()`.
- **`02_basic_binary_trait.R`** — Binary GBLUP with logit and probit links. Demonstrates new `predict(..., y_new = ...)` API, in-sample mode, `pred$prob` field, and 3-panel visualisation including a calibration plot. Uses `load_data()`.
- **`03_marker_QTL_congruency_theory.R`** — Comparison study: SNP vs MH under different QTL architectures, BayesR/BayesA/GBLUP/GWABLUP. Requires both `masreml` and `masbayes`. (Generates its own data; intentionally not migrated to `load_data()` because the example varies QTL architecture across runs.)
- **`04_gwas.R`** — EMMAX GWAS + GWABLUP showcase. SNP and MH paths fit on full `load_data()`, compare GBLUP vs GWABLUP heritability, and emit four Manhattan plots via `CMplot` using the bundled `d$map_snp` / `d$map_mh`. Mirror of `masbayes/examples/04_gwas.R`.

## Cross-package alignment with masbayes

`predict.masreml()` and `predict.masbayes_*()` now expose the same primary field names (`GEBV`, `prob`, `metrics`, `h2`, `eval_scope`, `has_truth`) so cross-package CV / comparison scripts can use a single accessor pattern. masreml retains `total_gebv` and `fitted` as aliases for backward compatibility. Both packages' binary metrics are on the observed/probability scale → directly comparable.

## Recent changes

- **v0.4.0 release** (2026-05-18) — maintenance release on top of 0.3.1; R API and FFI exports unchanged.
  - **Bug fix in `matrix/pedigree.rs`**: `build_a_ped_internal` now uses the correct direction of the Henderson (1976) off-diagonal recursion (`a_ij = 0.5 · (a_{j, sire_i} + a_{j, dam_i})` for `j < i`, i.e. parents of `i` rather than parents of `j`). The previous formula collapsed parent–offspring kinship to zero whenever the older individual was a founder with unknown parents, so all `a_ij` entries from `0.3.1` involving founders were wrong. Diagonal `a_ii` (inbreeding) was correct in 0.3.1 and is unchanged. Downstream impact: any pBLUP / A-based REML fit run under 0.3.1 against a pedigree containing founders should be re-run on 0.4.0; SNP-only and MH-only paths are unaffected.
  - **Rust module documentation**: every `.rs` file under `src/rust/src/` now has a `//!` module-level docstring covering algorithms, references, and design invariants. Picked up automatically by `masgenomics-docs/internals/` via `_scripts/extract-rust-docs.R`.
  - **Inline math comments** at every key algorithm: HE regression's four-step closed-form OLS derivation; AI-REML's Newton-style update with the AI matrix and step-halving safeguard; EM-REML's multiplicative non-negative update; EMMAX's per-marker Wald / LR test with the cached Cholesky factor.
  - **Function-level rustdoc** added to `matrix/snp_additive.rs` (`build_g_snp_add_internal`, `compute_allele_freq`, `center_w_vanraden`) and `matrix/pedigree.rs`.
  - **Rust unit-test layer** introduced: 22 tests under `#[cfg(test)]` (excluded from the release binary) covering Da encoding, frequency-weighted row shrinkage, VanRaden centering, pedigree A on a full-sib family with selfing, and dimension / error-path checks. All 22 tests pass.
  - **Internal rename** in `matrix/mh_additive.rs`: `project_sum_to_zero` → `frequency_weighted_row_shrinkage`. Math is identical to 0.3.1 (the previous name was misleading: the function partially damps the frequency-weighted row sum, it does not project it to zero). Bit-for-bit output parity verified by re-running the masbayes example 03 against `res.txt`.
- **GitHub Actions build-release workflow** (2026-05-16) — added `.github/workflows/build-release.yml` mirroring `masbayes/.github/workflows/build-release.yml` with adjustments for masreml's OpenBLAS dependency. Multi-OS matrix (Windows x64 via `openblas-static` + Rtools43, macOS Intel via `macos-13` + Apple Silicon via `macos-14` with `brew install openblas`, Linux x64 with `apt libopenblas-dev`), per-OS system-deps install, and a post-install smoke test (small AI-REML fit on bundled training split using `y_cont_qtl_snp` + `snp` markers) that gates the artefact upload — catches silent BLAS linking failures. Triggered on `push: tags: 'v*'` or `workflow_dispatch`. As of 2026-05-16 the workflow file sits on local branch `ci/build-release` (8 commits on top of `951eb88`) — not yet pushed, merged, or tagged. See **GitHub Actions release workflow** section above for the full pipeline description.
- **Auto post-fit summary** added to `masreml()` and `.masreml_binary()`; banner mirrors `masbayes::print_run_summary()`. `verbose`, `save_rds`, `save_path` arguments added (`save_rds = FALSE` by default — differs from masbayes because masreml is often called inside CV loops).
- **Training metrics** field (`fit$training_metrics`) added with R2/RMSE/accuracy/bias (continuous) or same + AUC (binary, observed scale). `summary.masreml` shows a Training Performance block.
- **`predict.masreml()` aligned with `predict.masbayes_*()`**: new args `y_new`, `X_new`; in-sample mode (`predict(fit)`); auto-compute metrics; primary field names `GEBV`/`prob` with backward-compat aliases `total_gebv`/`fitted`; carries `h2`/`sigma2`/`response_type`/`algorithm`/`eval_scope`/`has_truth`.
- **Binary metrics scale fix** — `evaluate_prediction()` and `predict()$metrics` for binary now compute on the observed (probability) scale so `bias` is the calibration slope (1.0 = ideal) instead of an artefact of mixed-scale comparison. AUC unchanged. Previously, `bias` was computed from `lm(y_01 ~ gebv_liability)` which produced spuriously low slopes.
- `gwablup()` calls `masreml()` internally with `verbose = FALSE` to avoid double banners.
