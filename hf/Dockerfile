# SPDX-License-Identifier: MIT OR Apache-2.0
# bankML on a Hugging Face Space (CPU): built from this repository's source, the model fetched at a pinned revision
# and verified by bankML's own gate (guard + sha256 pin) before it can answer.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential python3 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.cargo/bin:$PATH
WORKDIR /home/user/bankml
# Rust: the toolchain pinned in rust-toolchain.toml (1.99.0), nothing else
COPY --chown=user rust-toolchain.toml ./
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none \
    && rustup toolchain install "$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)" --profile minimal
# the model: PYTHAI's provenance fork of Ternary-Bonsai-8B at a fixed revision, with its FORK.json (the pin)
ARG MODEL_REPO=PYTHAI/Ternary-Bonsai-8B-gguf-fork
ARG MODEL_REV=2445960052d465ace262a43b5f5fef52a0c1e1ef
RUN mkdir -p models && for f in Ternary-Bonsai-8B-Q2_0_g64.gguf FORK.json; do \
      curl -sSfL -o "models/$f" "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/$f"; done
# the source, and the build: one binary, zero crates
COPY --chown=user . .
RUN cargo build --release --locked \
    && ./target/release/bankml verify models/Ternary-Bonsai-8B-Q2_0_g64.gguf --fork models/FORK.json
EXPOSE 7860
CMD ["sh", "hf/start.sh"]
