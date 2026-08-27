use std::{env, fs, io, path::PathBuf};

use pyo3_stub_gen::Result;
use symbolica_community::stub_info;

const FEYNKIT_MODULE: &str = "symbolica.community.feynkit";

fn normalize_stub_source(source: &str) -> String {
    let mut lines = source
        .lines()
        .map(|line| line.trim_end_matches([' ', '\t']))
        .collect::<Vec<_>>();
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    lines.join("\n") + "\n"
}

fn write_feynkit_stub() -> Result<()> {
    let info = feynkit_py::stub_info()?;
    let module = info.modules.get(FEYNKIT_MODULE).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "FeynKit did not contribute its public stub module",
        )
    })?;
    let source = normalize_stub_source(&module.to_string());

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python/symbolica/community/feynkit/__init__.pyi");
    fs::write(path, source)?;
    Ok(())
}

fn main() -> Result<()> {
    let feynkit_only = env::args_os().any(|argument| argument == "--feynkit-only");
    if !feynkit_only {
        let stub = stub_info()?;
        stub.generate()?;

        // `pyo3-stub-gen` treats leaf modules as adjacent `.pyi` files, while
        // this module is an importable package. Avoid leaving an ambiguous
        // second module next to the package after a full regeneration.
        let adjacent_stub = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python/symbolica/community/feynkit.pyi");
        if adjacent_stub.exists() {
            fs::remove_file(adjacent_stub)?;
        }
    }
    write_feynkit_stub()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::normalize_stub_source;

    #[test]
    fn normalizes_trailing_whitespace_and_end_of_file() {
        assert_eq!(
            normalize_stub_source("class Example:  \n    ...\t\n   \n\n"),
            "class Example:\n    ...\n"
        );
    }
}
