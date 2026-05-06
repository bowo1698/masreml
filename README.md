<a id="readme-top"></a>
# MasReml

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![R](https://img.shields.io/badge/R-4.4+-blue.svg)](https://www.r-project.org/)
[![Examples](https://img.shields.io/badge/Examples-Click%20Here-blue)](examples/)
[![Rust][Rust]][Rust-url]

**Universal REML-BLUP for SNP and Microhaplotype Genomic Prediction**

`masreml` is an R package with a Rust backend (via [extendr](https://extendr.github.io/)) for
fast and flexible genomic prediction using REML-BLUP. It supports SNP additive, SNP dominance,
microhaplotype (MH) additive, and pedigree-based relationship matrices, and handles both
continuous and binary traits.

---

## Features

- **Multiple marker types**: SNP additive (VanRaden 2008), SNP dominance (Da et al. 2014),
  microhaplotype additive (Da 2015 W_αh coding), pedigree-based A matrix
- **Multiple REML algorithms**: HE regression, AI-REML, EM-REML, HE-initialized AI-REML (HI)
- **Binary trait support**: single-step Laplace approximation on liability scale (logit or probit link)
- **Flexible solvers**: Cholesky (small n) and PCG (large n) for EBV
- **Multi-component models**: simultaneous fitting of additive + dominance, or SNP + MH
- **GWAS**: EMMAX-based genome-wide association study via `run_gwas()`
- **GWABLUP**: GWAS-assisted genomic prediction via `gwablup()` (Meuwissen et al. 2024)
- **Cross-validation**: built-in k-fold CV via `cv_masreml()`
- **Fast Rust backend**: core linear algebra implemented in Rust for performance

---

## Requirements

`masreml` requires **Rust** to compile the backend. Install Rust via [rustup](https://rustup.rs/):

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Verify
rustc --version
cargo --version
```

On Windows, download and run the installer from https://rustup.rs.

R package dependencies: `devtools`, `MASS` (for examples).

---

## Installation

```r
# Install from GitHub
devtools::install_github("bowo1698/masreml")
```

Or from a local clone:

```bash
git clone https://github.com/bowo1698/masreml.git
cd masreml
```

```r
devtools::install()
```

### Remove

```r
remove.packages("masreml")
```

To fully remove including compiled artifacts:

```bash
# Remove installed package
Rscript -e 'remove.packages("masreml")'

# Remove build artifacts (from source directory)
cd masreml/src/rust
cargo clean
```

---

## Quick Start

### Continuous Trait — SNP GBLUP

```r
library(masreml)

# Simulate data
set.seed(42)
n <- 500; m <- 300
W <- matrix(sample(0:2, n * m, replace = TRUE), n, m)
rownames(W) <- paste0("ind", 1:n)

G  <- build_G_snp(W)
u  <- MASS::mvrnorm(1, rep(0, n), G * 0.4)
y  <- u + rnorm(n, 0, sqrt(0.6))
names(y) <- rownames(W)

# Fit GBLUP
fit <- masreml(y = y, markers = list(snp_add = W))
summary(fit)

# Accuracy
compute_accuracy(fit$total_gebv, y, h2 = fit$varcomp$h2["snp_add"])
```

### Binary Trait

```r
# Binary phenotype (0/1)
liability <- u + rnorm(n, 0, sqrt(0.7))
y_bin     <- as.integer(liability > quantile(liability, 0.6))
names(y_bin) <- rownames(W)

fit_bin <- masreml(
  y       = y_bin,
  markers = list(snp_add = W),
  trait   = "binary",
  link    = "logit"
)
summary(fit_bin)
# fit_bin$binary$auc        — AUC
# fit_bin$binary$h2_liability — h2 on liability scale
```

### Multi-component Model (SNP additive + dominance)

```r
fit_ad <- masreml(
  y       = y,
  markers = list(snp_add = W, snp_dom = W),
  method  = "AI"
)
summary(fit_ad)
varcomp(fit_ad)
```

### GWAS and GWABLUP

```r
# Step 1: fit standard GBLUP
fit <- masreml(y, markers = list(snp_add = W))

# Step 2: run GWAS
gwas <- run_gwas(
  markers     = list(snp_add = W),
  y           = y,
  masreml_fit = fit
)
summary(gwas)

# Step 3: GWAS-assisted prediction
fit_wa <- gwablup(y, markers = list(snp_add = W), gwas_result = gwas)
summary(fit_wa)
```

### Microhaplotype (MH) GBLUP

```r
# W_mh: W_αh matrix (Da 2015 coding) from construct_wah_matrix() in masbayes
fit_mh <- masreml(
  y       = y,
  markers = list(mh_add = W_mh)
)
summary(fit_mh)
```

### Cross-validation

```r
cv <- cv_masreml(
  y       = y,
  markers = list(snp_add = W),
  folds   = 5,
  method  = "auto"
)
summary(cv)
```

### Pre-built G matrix

```r
G   <- build_G_snp(W)
fit <- masreml(y = y, G = list(snp_add = G))
```

---

## Main Functions

| Function | Description |
|---|---|
| `masreml()` | Main REML-BLUP interface |
| `run_gwas()` | EMMAX GWAS for SNP or MH markers |
| `gwablup()` | GWAS-assisted genomic prediction |
| `build_G_snp()` | SNP additive G matrix (VanRaden 2008) |
| `build_D_snp()` | SNP dominance D matrix |
| `build_G_mh()` | MH additive G matrix (Da 2015) |
| `build_A_ped()` | Pedigree-based A matrix |
| `cv_masreml()` | k-fold cross-validation |
| `compute_accuracy()` | Prediction accuracy (r, slope, r_MG) |
| `varcomp()` | Extract variance components |
| `summary()` | Model summary |

---

## REML Methods

| Method | Description | Recommended for |
|---|---|---|
| `"auto"` | AI-REML for continuous, HE for binary | Default |
| `"HE"` | Haseman-Elston regression (fast, 1 step) | Large n, binary |
| `"AI"` | Average Information REML (accurate) | Continuous, moderate n |
| `"EM"` | EM-REML (stable, slow) | When AI diverges |
| `"HI"` | HE-initialized AI-REML (fast + accurate) | Best overall for continuous |

---

## License

GPL-3 License - see [LICENSE](LICENSE) file

Copyright (c) 2025 Agus Wibowo

---

## Contact

- **Email**: aguswibowo1698@gmail.com

---

## References

- VanRaden PM. Efficient methods to compute genomic predictions. [J. Dairy Sci. 91,4414–4423 (2008)](https://doi.org/10.3168/jds.2007-0980)

- Da, Y. Multi-allelic haplotype model based on genetic partition for genomic prediction and variance component estimation using SNP markers. [BMC Genet. 16, 144 (2015)](https://doi.org/10.1186/s12863-015-0301-1)

- Johnson DL, Thompson R. Restricted maximum likelihood estimation of variance components for univariate animal models. [J. Dairy Sci. 78, 449–456 (1995)](https://doi.org/10.3168/jds.S0022-0302(95)76654-1)

- Meuwissen THE et al. (2024) GWABLUP: genome-wide association assisted best linear unbiased prediction of genetic values. [Genet Sel Evol. 56,17 (2024)](https://doi.org/10.1186/s12711-024-00881-y)

- Kang HM et al. (2010) Variance component model to account for sample structure in genome-wide association studies. [Nat Genet 42, 348–354 (2010)](https://doi.org/10.1038/ng.548)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Development Team

**Lead Developer:** Agus Wibowo  
James Cook University

**Supervisors:**  
- Prof. Kyall Zenger
- Dr. Cecile Massault

---

<p align="center">
  <strong>masreml</strong> - Making reliable and faster genomic prediction🧬
</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Rust]: https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white
[Rust-url]: https://rust-lang.org/