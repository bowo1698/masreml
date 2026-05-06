<a id="readme-top"></a>
# MasReml

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![R](https://img.shields.io/badge/R-4.4+-blue.svg)](https://www.r-project.org/)
[![Examples](https://img.shields.io/badge/Examples-Click%20Here-blue)](examples/)
[![Rust][Rust]][Rust-url]

**Universal REML-BLUP for biallelic SNP and multi-allelic microhaplotype genomic prediction**

`masreml` implements **REML-BLUP genomic prediction** for both bi-allelic SNP and multi-allelic microhaplotype markers. The package supports four relationship matrix types, multiple REML algorithms, and binary traits, with all numerical computation handled by a Rust backend.

---

## Relationship Matrices

**SNP additive G (VanRaden 2008)**

$$G = \frac{W W'}{k}, \quad k = \sum_j 2p_j(1-p_j)$$

where $W_{ij} = X_{ij} - 2p_j$ is the allele-frequency-centered genotype matrix and $p_j$ is the allele frequency at locus $j$.

**SNP dominance D (Da et al. 2015)**

$$D = \frac{W_\delta W_\delta'}{k_\delta}, \quad k_\delta = \frac{\text{tr}(W_\delta W_\delta')}{n}$$

where the dominance coding captures heterozygote deviation from additive expectation:

$$w_{\delta,ij} = \begin{cases} -2p_j^2 & X_{ij} = 0 \text{ (AA)} \\ 2p_j(1-p_j) & X_{ij} = 1 \text{ (Aa)} \\ -2(1-p_j)^2 & X_{ij} = 2 \text{ (aa)} \end{cases}$$

**Microhaplotype additive $A_{gh}$ (Da 2015)**

$$A_{gh} = \frac{\sum_\alpha W_\alpha W_\alpha'}{k_{gh}}, \quad k_{gh} = \frac{\text{tr}\!\left(\sum_\alpha W_\alpha W_\alpha'\right)}{n}$$

where $\alpha$ indexes haplotype blocks. For each block, the W_αh matrix is built by dropping the most frequent allele and applying Da (2015) coding for each retained allele $k$:

$$w_{\alpha h,ik} = \begin{cases} 2p_{\alpha k} & \text{individual } i \text{ does not carry allele } k \\ -(1 - 2p_{\alpha k}) & \text{individual } i \text{ heterozygous for } k \\ -2(1 - p_{\alpha k}) & \text{individual } i \text{ homozygous for } k \end{cases}$$

A sum-to-zero constraint $\sum_k p_{\alpha k} \cdot w_{i\alpha k} = 0$ is enforced per individual per locus, ensuring orthogonal allele effect decomposition. Each block contributes $n_\alpha - 1$ allele dimensions to the relationship matrix.

**Pedigree A matrix (Henderson 1976)**

Numerator relationship matrix built via Henderson's recursive tabular algorithm:

$$a_{ii} = 1 + F_i, \qquad F_i = \begin{cases} \frac{1}{2}a_{s_i, d_i} & \text{both parents known} \\ 0 & \text{otherwise} \end{cases}$$

$$a_{ij} = \frac{1}{2}(a_{i,\, s_j} + a_{i,\, d_j}), \quad i > j$$

where $s_i$ and $d_i$ are the sire and dam indices of individual $i$, and $F_i$ is the inbreeding coefficient. The matrix is computed in a single forward pass assuming individuals are ordered such that parents precede offspring. Unknown parents contribute zero to the relationship.

## Linear Model

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u} + \mathbf{e}, \quad \mathbf{u} \sim \mathcal{N}(\mathbf{0},\ \mathbf{G}\sigma^2_g), \quad \mathbf{e} \sim \mathcal{N}(\mathbf{0},\ \mathbf{I}\sigma^2_e)$$

Multi-component models extend to $\mathbf{V} = \sum_k \mathbf{G}_k \sigma^2_{g_k} + \mathbf{I}\sigma^2_e$, to enable simultaneous partitioning of additive + dominance, or SNP + microhaplotype variance components.

Variance components are estimated by REML; GEBV per component are solved from the mixed model equations (MME): $\hat{\mathbf{u}}_k = \mathbf{G}_k \mathbf{V}^{-1}(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})\,\sigma^2_{g_k}$.

For binary traits (0/1), a single-step Laplace approximation is applied on the liability scale using a working response derived from the logit or probit link, with heritability transformed via the Dempster-Falconer formula.

## REML Algorithms

| Method | Algorithm | Best for |
|--------|-----------|----------|
| `"HI"` | HE-initialized AI-REML | Continuous trait, general use |
| `"AI"` | Average Information REML (Gilmour et al. 1995) | Accurate, moderate n |
| `"HE"` | Haseman-Elston regression (fast, one-step) | Large n, binary trait |
| `"EM"` | EM-REML (Meyer 1989) | When AI diverges |
| `"auto"` | AI for continuous, HE for binary | Default |

## Solvers

GEBV are solved via direct **Cholesky factorization** (n < 10,000) or **Preconditioned Conjugate Gradient** (n ≥ 10,000). A single Cholesky factorization of $\mathbf{V}$ is shared across REML gradient computation, BLUP solving, and EMMAX GWAS, to avoid redundant $O(n^3)$ operations.

## Additional features

- **Flexible solvers**: Cholesky (small n) and PCG (large n) for EBV
- **GWAS**: EMMAX-based genome-wide association study via `run_gwas()`
- **GWABLUP**: GWAS-assisted genomic prediction via `gwablup()` (Meuwissen et al. 2024)

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

R package dependencies: `devtools` and `MASS`.

---

## Installation

```r
# Install from GitHub
devtools::install_github("bowo1698/masreml")
```

---

## Code examples

### Build Relationship Matrices

```r
library(masreml)

# SNP additive G (VanRaden 2008)
# W: n x m integer matrix, values 0/1/2
G_add <- build_G_snp(W)

# MH additive Agh (Da 2015)
# mh_list: list of data.frames, one per chromosome
# cols: ID, strand1_block1, strand2_block1, strand1_block2, ...
A_mh <- build_G_mh(mh_list)

# SNP dominance D 
D_dom <- build_D_snp(W)

# OR Training-based allele frequencies (prevents data leakage)
G_snp_full <- build_G_snp(W_all, ref_W = W_train)
G_mh_full  <- build_G_mh(hap_all, ref_mh = hap_train, ids = all_ids)
```

### Model fitting

```r
# SNP additive GBLUP (continuous trait) with automatic REML algorithm selection
fit <- masreml(y, markers = list(snp_add = G_add), method = "auto")
summary(fit)

# MH additive GBLUP (continuous trait)
fit_mh <- masreml(y, markers = list(mh_add = A_mh), method = "auto")

# Full effect model: SNP additive + SNP dominance + MH additive
fit <- masreml(
  y = y,
  G = list(
    snp_add = G_add,
    snp_dom = D_dom,
    mh_add  = A_mh
  ),
  method = "auto" # or AI, EM, HE, HI
)

# Binary trait (logit link, HE-REML)
fit_bin <- masreml(y, markers = list(snp_add = G_add), trait = "binary", method = "auto")
```

### Inspect results

```r
summary(fit)

# Variance components
varcomp(fit)
#   Component   Sigma2     H2 Proportion
#     snp_add 0.412300 0.4123     0.4123
#    residual 0.587700     NA     0.5877

# GEBV
head(fit$total_gebv)
head(fit$gebv$snp_add)
```

### Predict new individuals

```r
# Mode A: pre-built full G matrix (n_total x n_total)
G_full <- build_G_snp(W_all)
pred <- predict(fit,
  G_full    = list(snp_add = G_full),
  train_ids = rownames(W_train),
  test_ids  = rownames(W_test))

# Mode B: raw markers for test set
pred <- predict(fit,
  markers_new   = list(snp_add = W_test),
  markers_train = list(snp_add = W_train))

pred$total_gebv
```

### Evaluate accuracy

```r
acc <- compute_accuracy(
  gebv = fit$total_gebv,
  y    = y,
  h2   = fit$varcomp$h2["snp_add"]
)
#      r  slope   r_MG
# 0.823  0.991  0.914
```

### 6. GWAS and GWABLUP

```r
# EMMAX GWAS
gwas_res <- run_gwas(
  markers     = list(snp_add = W_train),
  y           = y_train,
  masreml_fit = fit,
  ref_markers = list(snp_add = W_train)
)
summary(gwas_res)

# GWAS-assisted prediction
fit_wa <- gwablup(
  y           = y_train,
  markers     = list(snp_add = W_train),
  gwas_result = gwas_res
)
summary(fit_wa)
```

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