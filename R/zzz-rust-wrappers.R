# Rust entry points — internal use only

#' @noRd
r_build_g_snp_add <- function(w) .Call(wrap__r_build_g_snp_add, w)

#' @noRd
r_build_g_snp_dom <- function(w) .Call(wrap__r_build_g_snp_dom, w)

#' @noRd
r_build_g_mh_add <- function(hap1, hap2, n_alleles) .Call(wrap__r_build_g_mh_add, hap1, hap2, n_alleles)

#' @noRd
r_build_a_ped <- function(sire, dam, n) .Call(wrap__r_build_a_ped, sire, dam, n)

#' @noRd
r_run_reml <- function(y, x, g_list, method, max_iter, tol, n_threads) .Call(wrap__r_run_reml, y, x, g_list, method, max_iter, tol, n_threads)

#' @noRd
r_solve_ebv <- function(y, x, g_list, sigma2, solver, max_iter, tol) .Call(wrap__r_solve_ebv, y, x, g_list, sigma2, solver, max_iter, tol)