use std::process::Command;

fn main() {
    // Re-run if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");

    let pkg_version = env!("CARGO_PKG_VERSION");

    // Allow the version to be injected via HAR_BUILD_VERSION env var
    // (useful when building inside containers without git)
    if let Ok(v) = std::env::var("HAR_BUILD_VERSION") {
        println!("cargo:rustc-env=HAR_VERSION={}", v);
        return;
    }

    // Check if HEAD is exactly tagged
    let tagged = Command::new("git")
        .args(["describe", "--tags", "--exact-match", "HEAD"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let version = if tagged {
        pkg_version.to_string()
    } else {
        let sha = Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "unknown".to_string());
        format!("{}+g{}", pkg_version, sha)
    };

    println!("cargo:rustc-env=HAR_VERSION={}", version);
}
