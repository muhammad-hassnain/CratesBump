# Sandbox image for scripts/label_audit_cargo_semver_checks.py.
#
# The audit builds randomly sampled crates from crates.io, which runs their
# build scripts and proc macros. The script therefore runs every build in a
# throwaway container from this image: non-root, no capabilities, no network
# during the build (dependencies are fetched in a separate, build-free step),
# and no host files mounted except the pair's own scratch directory.
#
#   docker build -t cratesbump-audit:0.51.0 -f docker/audit.Dockerfile docker/
FROM rust:1.99.1-bookworm

# System libraries that common -sys crates link against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      cmake clang libclang-dev protobuf-compiler pkg-config libssl-dev \
      libsqlite3-dev libpq-dev libzstd-dev libudev-dev libdbus-1-dev \
      libfontconfig1-dev libxcb1-dev libx11-dev libasound2-dev \
 && rm -rf /var/lib/apt/lists/*

RUN cargo install cargo-semver-checks --version 0.51.0 --locked \
 && rm -rf /usr/local/cargo/registry /usr/local/cargo/git

RUN useradd -m -u 1000 audit
USER audit
ENV CARGO_TERM_COLOR=never
WORKDIR /work
