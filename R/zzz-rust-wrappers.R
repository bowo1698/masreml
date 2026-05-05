# Rust entry points — internal use only

#' @noRd
r_build_g_snp_add <- function(w, weights = NULL, allele_freq = NULL) .Call(wrap__r_build_g_snp_add, w, weights, allele_freq)

#' @noRd
r_build_g_snp_dom <- function(w) .Call(wrap__r_build_g_snp_dom, w)

#' @noRd
r_build_g_mh_add <- function(hap1, hap2, n_alleles, weights = NULL, ref_hap1 = NULL, ref_hap2 = NULL) .Call(wrap__r_build_g_mh_add, hap1, hap2, n_alleles, weights, ref_hap1, ref_hap2)

#' @noRd
r_build_a_ped <- function(sire, dam, n) .Call(wrap__r_build_a_ped, sire, dam, n)

#' @noRd
r_run_reml <- function(y, x, g_list, method, max_iter, tol, n_threads) .Call(wrap__r_run_reml, y, x, g_list, method, max_iter, tol, n_threads)

#' @noRd
r_solve_ebv <- function(y, x, g_list, sigma2, solver, max_iter, tol) .Call(wrap__r_solve_ebv, y, x, g_list, sigma2, solver, max_iter, tol)

#' @noRd
r_run_emmax_snp <- function(w, y, x, sigma2_g, sigma2_e, g_u) {
    .Call(wrap__r_run_emmax_snp, w, y, x, sigma2_g, sigma2_e, g_u)
}

#' @noRd
r_run_emmax_mh <- function(hap1, hap2, n_alleles, y, x, sigma2_g, sigma2_e, g_u) {
    .Call(wrap__r_run_emmax_mh, hap1, hap2, n_alleles, y, x, sigma2_g, sigma2_e, g_u)
}
#' @noRd
r_smooth_and_pp <- function(lr, window, pi) {
    .Call(wrap__r_smooth_and_pp, lr, window, pi)
}