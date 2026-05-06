fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .unwrap_or_default();

    match target_os.as_str() {
        "macos" => {
            // Homebrew OpenBLAS path (Intel dan Apple Silicon)
            let homebrew_prefix = if cfg!(target_arch = "aarch64") {
                "/opt/homebrew"  // Apple Silicon
            } else {
                "/usr/local"     // Intel Mac
            };

            println!(
                "cargo:rustc-link-search=native={}/opt/openblas/lib",
                homebrew_prefix
            );
            println!("cargo:rustc-link-lib=openblas");
            println!(
                "cargo:rustc-env=OPENBLAS_DIR={}/opt/openblas",
                homebrew_prefix
            );
        }

        "linux" => {
            // Standard Linux paths
            // apt: libopenblas-dev
            // yum/dnf: openblas-devel
            println!("cargo:rustc-link-lib=openblas");

            // Common Linux OpenBLAS locations
            for path in &[
                "/usr/lib",
                "/usr/lib64",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/aarch64-linux-gnu",  // ARM Linux (HPC)
            ] {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }

        "windows" => {
            // Windows: use openblas-static (compile from source)
            // or via vcpkg: vcpkg install openblas
            if let Ok(vcpkg_root) = std::env::var("VCPKG_ROOT") {
                println!(
                    "cargo:rustc-link-search=native={}/installed/x64-windows/lib",
                    vcpkg_root
                );
                println!("cargo:rustc-link-lib=openblas");
            }
            // Fallback: openblas-static handles it automatically
        }

        _ => {
            println!("cargo:warning=Unknown OS, trying default OpenBLAS linking");
        }
    }

    // HPC: respect OPENBLAS_DIR env variable if set by user/module system
    // e.g. module load openblas → sets OPENBLAS_DIR
    if let Ok(openblas_dir) = std::env::var("OPENBLAS_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", openblas_dir);
        println!("cargo:rustc-link-lib=openblas");
    }
}