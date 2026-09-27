# Modeling — Agent Guidelines

**Stack:** Rust-based Machine Learning exploration repository powered by the `candle` framework. Focused on deep learning, generative AI, and personal model fine-tuning with native GPU acceleration on Apple Silicon (Metal). 

## 🚧 Status (Experimentation & Learning)
- **Fast Iteration:** This is an active ML modeling repository. Do not write migration paths, dual-writes, deprecation shims, or versioned schemas. Focus on clear, correct mathematical representations and model architectures.
- **Ask First (External/Deps):** Require explicit approval before any dependency change (e.g. editing `Cargo.toml`). Version bumps or lockfile-only refreshes (`cargo update`) count as dependency changes.
- **Quality Gate:** Keep the build passing. Never commit code that breaks cargo checks.

## 🗺️ Where Things Live
| Path | What |
|---|---|
| `src/bin/` | Executable binaries for individual ML experiments and model implementations (e.g., `llm.rs` for generative models, `habits.rs` for custom networks). |
| `src/` | Shared libraries, dataset loaders, or custom NN layers. |
| `target/` | Cargo build cache and compiled binaries. |

## 🛠️ Verification & Commands
- **Run by Default After Every Change (Static Checks):** No permission needed. Run the ones covering what you touched before reporting, and report any failure verbatim.
  - Rust: `cargo check`
- **Run ONLY When Explicitly Asked:**
  - Never run `cargo run` without permission, as downloading heavy tensor weights (e.g., from Hugging Face) or running high-epoch training loops can consume massive bandwidth/compute.
  - If unprompted to run the model, state that the implementation is unverified.

## 🏗️ Architecture & Code Rules
- **Rust (Source of Truth):**
  - Always utilize `candle_core::utils::metal_is_available()` where possible to leverage the workstation's native Apple Silicon (Metal) acceleration. Default to `.new_metal(0)` when initializing devices.
  - ML code should favor explicitly named tensor dimensions where possible to prevent shape mismatches during forward/backward passes.
  - Keep models lightweight to accommodate local inference. Consider quantization patterns (e.g., GGUF format via `candle-transformers`) when deploying LLMs.
- **Agent Hygiene:**
  - Never modify `/AGENTS.md` unless the user asks outright; propose changes as a diff. 
  - No obvious comments (document *why*, never *what*). No AI signatures in code or commits.
  - **No Scripted Edits:** Never use Python, shell, Node, or regex scripts to bulk-edit any tracked file: code, comments, docs (Markdown/MDX), SQL, or config (JSON, YAML, TOML). Edit each file deliberately. 

## 📝 Git Rules (Strict)
- **Commit Only on Command:** Never run `git add` or `git commit` unless explicitly instructed.
- **Format:** Conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`). **Hard limit of 1–5 lines** (1 subject line + optional blank line + max 3 body lines explaining *why*). No file lists, test outputs, or task summaries.
- **No Branches or Pushes:** Never create branches (`checkout -b`, `switch -c`, `branch`, or worktrees)—commit directly to the current branch. Never run `git push`.

## ✅ Definition of Done
Before reporting a change as finished:
- [ ] Static checks pass (`cargo check`), or the failure is reported verbatim.
- [ ] Ensure any newly added `.safetensors`, `.gguf`, or weighty datasets are properly `.gitignore`d so they aren't accidentally pushed to git.
- [ ] No `git add` or `git commit` unless asked, and when asked, a conventional commit of 1–5 lines. No branches, worktrees or pushes.
- [ ] No unapproved dependency changes.
