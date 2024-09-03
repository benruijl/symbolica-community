<h1 align="center">
  <br>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://symbolica.io/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://symbolica.io/logo.svg">
  <img src="https://symbolica.io/logo.svg" alt="logo" width="200">
</picture>
  <br>
</h1>

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://reform.zulipchat.com"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/benruijl/symbolica_community"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
</p>

# Community-enhanced Symbolica 

This repository contains a community-enhanced [Symbolica](https://github.com/benruijl/symbolica) library. It adds extra functionality to existing Symbolica types and defines new ones.

For example, a community function that computes a Dirac trace, can be injected into a Symbolica `Expression`:

```python
Expression.parse("gamma(mu1,mu2,mu3,mu4)").trace()
```


## Usage

The community-enhanced version is currently in the testing stage and requires manual building. In the near future, it will be accessible through `pip` and `crates.io`.


### Python
To use the extension in Python, use `maturin` to build it:

```bash
maturin build --release
```

### Rust

If you want to use Symbolica as a library in Rust, simply include it in the `Cargo.toml`:

```toml
[dependencies]
symbolica_community = { git = "https://github.com/benruijl/symbolica_community.git" }
```


## Contributing

Users can easily contribute Python or Rust code that extends Symbolica's functionality via Pull Requests. All code in this repository is MIT licensed.

Pure Python contributions should go in the `python/symbolica_community/[category]` folder, where the `category` is for example `physics`, `chemisty`, etc. All code must be fully typed.

Rust contributions go in `src/[category]/myfeature.rs` and potential Python bindings need to be registered in `src/[category].rs`

## Forward compatibility

All code accepted into the repository will continuously be upgraded to the latest stable version of Symbolica by Ruijl Research (potentially in combination with other contributors).