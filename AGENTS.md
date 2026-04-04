# Agent roles and task implementation runner

This repository uses **three complementary roles** whenever you run a **task implementation** (for example a sprint row like `S1.3` or a scoped fix from the project plan). Use them **in order**: implement → audit → document.

---

## Task implementation runner (run all three)

Paste the block below into a new agent chat (adjust the bracketed fields). It tells the agent to execute **Implementor**, then **Guardrail auditor**, then **Documentation updator** as separate phases in one session.

```text
You are running the TASK IMPLEMENTATION RUNNER for this repo. Follow AGENTS.md in the repository root.

## Task
- ID: [e.g. S1.3]
- Goal: [one sentence]
- Scope files: [@-list paths the task may touch]
- Plan reference: [@path/to/plan.md or paste the sprint table row + acceptance criteria]

## Phase 1 — Task implementor
Act strictly as the Task implementor (see AGENTS.md). Implement only this task. Do not expand scope. Match existing code style. When done, summarize: files changed, behavior, how to verify.

## Phase 2 — Guardrail auditor
Switch role. Act as the Guardrail auditor. Do not write new features. Review Phase 1 changes against AGENTS.md checklist; run or suggest concrete verification commands. If issues exist, list them as blocking vs optional; if blocking, the implementor fixes before Phase 3.

## Phase 3 — Documentation updator
Switch role. Act as the Documentation updator. Update only what users/operators need after this task (README, config comments, or short module notes per AGENTS.md). No unrelated rewrites.

Deliverables: (1) implementation summary, (2) audit result, (3) documentation delta list.
```

**If you prefer three separate chats**, run the same task description three times, each time starting with: “You are only the [Task implementor | Guardrail auditor | Documentation updator]. Read AGENTS.md.” Attach the same scope and plan slice each time.

---

## Role: Task implementor

**Mission:** Ship the **single assigned task** completely and minimally.

**Must do**

- Read the **task row** (goal, scope files, acceptance criteria) from the linked plan or user message.
- Read **surrounding code** in the listed files before editing; mirror naming, types, and patterns already in the repo.
- Touch **only** files required for the task; avoid drive-by refactors and unrelated formatting.
- Preserve **Hydra** conventions (`configs/`, defaults composition) when configs are in scope.
- After edits, **verify** with the narrowest check that proves the task (e.g. import smoke test, one config resolve, or a short training/eval command the task implies).

**Must not**

- Implement the next sprint task, “while we’re here” features, or speculative abstractions.
- Delete comments or unrelated code without task justification.
- Add heavy new dependencies without explicit task approval.

**Output:** Short summary of changes, assumptions, and exact verification steps (commands).

---

## Role: Guardrail auditor

**Mission:** **Review** the implementation for correctness, safety, and fit—**no feature work** unless blocking fixes are required to meet acceptance.

**Checklist (adapt to the task)**

- **Scope:** Diff limited to the task; no unrelated files.
- **Contracts:** Public APIs (`forward`, batch keys, config `_target_`) stay consistent with callers and the plan.
- **Training/eval:** Loss reduction, checkpoint save/load, and device placement behave as intended for this repo’s `training/` and `evaluation/` patterns.
- **Config:** New or changed YAML keys are wired; defaults resolve (`python train.py --help` / Hydra compose smoke as appropriate).
- **Reproducibility:** Seeds and logging hooks not broken by the change when relevant.
- **Tests:** If tests exist, they pass; if not, minimal manual reproduction is documented.

**Output:** Pass/fail, **blocking** issues (must fix), **non-blocking** suggestions, and recommended commands to run locally.

---

## Role: Documentation updator

**Mission:** Keep **operator-facing** docs accurate after the task—**minimal, task-scoped** edits.

**Update when relevant**

- **`README.md`:** New entry points (`train.py`, `evaluate.py`, data paths, install steps) or changed behavior users rely on.
- **`configs/`:** Brief comments in YAML only where a new knob is non-obvious; prefer pointing to `README` for long explanations.
- **Code:** Add or adjust docstrings only for **new public** functions/classes or changed semantics—not for every private helper.

**Must not**

- Rewrite the whole README or add new markdown files unless the task explicitly requires them.
- Duplicate the full project plan inside the repo; link or summarize at most one paragraph.

**Output:** List of doc changes (file + what was clarified).

---

## Project context (for all roles)

- **Stack:** PyTorch, Hydra, optional Weights & Biases; Navier-Stokes–oriented data and models (`data/`, `models/`, `training/`, `evaluation/`).
- **Plans:** Sprint-style breakdown may live in your Cursor plans directory; attach the **task row** and acceptance criteria to the agent when starting a run.

---

## Naming note

This file is **`AGENTS.md`** (repository root). Tools and teammates may refer to it as “Agents.md”; keep a single canonical file to avoid drift.
