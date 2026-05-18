//! Shared utilities.
//!
//! Currently a single submodule, [`linalg`], collecting small linear
//! algebra primitives that are used across [`crate::matrix`],
//! [`crate::reml`], [`crate::solver`], and [`crate::gwas`].
//!
//! Kept as a top-level module so future additions (RNG helpers,
//! benchmarking shims, etc.) have an obvious home without disturbing the
//! existing public surface.

pub mod linalg;