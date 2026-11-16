// Every library crate that can be published carries the crates.io metadata keys
// and a README next to its manifest. Only `std` is used.

use std::fs;
use std::path::PathBuf;

const CRATES: [&str; 8] = [
    "kit-core",
    "kit-http",
    "arraykit",
    "frame",
    "scikit",
    "plotkit",
    "agent",
    "agent-derive",
];

const REQUIRED: [&str; 4] = [
    "description",
    "license.workspace = true",
    "repository.workspace = true",
    "readme = \"README.md\"",
];

#[test]
fn all_library_crates_have_publish_metadata() {
    let crates_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut problems = Vec::new();

    for name in CRATES {
        let dir = crates_dir.join(name);
        let manifest = match fs::read_to_string(dir.join("Cargo.toml")) {
            Ok(text) => text,
            Err(err) => {
                problems.push(format!("{name}: cannot read Cargo.toml: {err}"));
                continue;
            }
        };
        for key in REQUIRED {
            if !manifest.contains(key) {
                problems.push(format!("{name}: Cargo.toml has no `{key}`"));
            }
        }
        if !dir.join("README.md").is_file() {
            problems.push(format!("{name}: no README.md next to Cargo.toml"));
        }
    }

    assert!(
        problems.is_empty(),
        "publish metadata missing:\n{}",
        problems.join("\n")
    );
}
