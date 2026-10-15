# SPDX-License-Identifier: MIT OR Apache-2.0
"""gitmind — git state monitor + multi-source backup/rollback. MIT licensed.

Vendored in bankml from github.com/Professor-Codephreak/gitmind 0.1.0 (commit c4b379c), unchanged but for the SPDX headers
(same authors: MIT OR Apache-2.0 here, as LICENSING.md says); its MIT licence is
LICENSE beside this file. bankml uses it through sAGI/memgit.py, for .memory versions, with no backup sources.

Model-agnostic infrastructure: no LLM dependency, works the same whatever model
you run (GLM-4.5/5.2, Llama, Qwen, Claude, …).
"""
from .core import GitMind, Source, LocalSource, IPFSSource

__version__ = "0.1.0"
__all__ = ["GitMind", "Source", "LocalSource", "IPFSSource"]
