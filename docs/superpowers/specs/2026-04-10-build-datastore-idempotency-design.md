# Build MLX Datastore — Idempotency Design

## Problem

`build_mlx_datastore.sh` has four substantive phases (CHAINS prep, clean-speech
preprocessing, background-music prep, audio cache build). Each takes minutes to
hours. When the script crashes mid-run (e.g., out of space) and is re-launched,
it re-enters every phase from the top. Phases 1–3 do have per-file resume
internally (skip individual WAVs that already exist), but the bash script still
calls each Python script unconditionally — incurring startup, scanning, and
model-loading overhead even when the phase already completed in a prior run.

The user's constraint: **lists and indices can be stale** (e.g., a crash leaves
a partial list on disk). The only trustworthy evidence of completion is the
**actual output files** (WAVs for prep phases, NPZ shards for the cache build).

## Proposed Approach: Disk-Verified Phase Skipping

Before calling each phase's Python script, run a **fast verification function**
in the bash script that checks actual output files on disk. If the check passes,
skip the phase entirely with a clear log message. If it fails, run the phase
normally (existing per-file resume handles incremental work).

### Design Principles

1. **Files/shards are truth** — never skip based solely on a list file existing.
   Validate that referenced files are present on disk **and have size > 0**.
2. **No thresholds** — a crash at 96% must not cause a permanent skip. Use
   exact checks: either every expected output exists, or the phase re-runs.
3. **Conservative by default** — when in doubt, re-run the phase. The internal
   resume makes re-running a complete phase cheap (just scanning overhead).
4. **Merged lists always regenerated** — even when a producer phase is skipped,
   its downstream merged list must be rebuilt from current raw lists.
5. **Transparent** — log exactly why a phase was skipped or why it's re-running.
6. **Single `--force` flag** — disables all phase-level skip checks.

### Approaches Considered

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. Completion sentinels** | Write a `.done` marker after each phase succeeds; check marker mtime vs input mtime | Fast, simple | Yet another metadata file that can be stale; doesn't verify actual data on disk |
| **B. Disk-verified skip (chosen)** | Before each phase, validate output files exist on disk via phase-specific Python checks | Trustworthy, handles crashes gracefully, no sentinel management | Slightly more code; full validation scan |
| **C. List regeneration** | Scan output directories, regenerate lists from disk, compare to inputs | Most thorough | Scanning large dirs is slow; complex path-reconstruction logic |

**Decision:** Approach B. It directly checks what the user cares about (do the
output files actually exist on disk?) without introducing new metadata files or
requiring expensive directory scans.

### Limitation: Parameter Changes

This design does **not** detect when phase parameters change between runs
(e.g., different `--preprocess-model`, `--music-prepare-style`, or
`--sample-rate`). If you change parameters, use `--force` or the
phase-specific `--*-overwrite` flag to force a rebuild.

Adding parameter fingerprinting (hashing current options and comparing to a
stored fingerprint) was considered but rejected as disproportionate complexity
for a rare edge case. The user knows when they change parameters.

---

## Phase-by-Phase Design

### Shared Helper: `verify_file_list()`

A Python-based function embedded in the bash script that validates a file list
against disk reality:

```
verify_file_list <list_file>
```

**Logic:**
1. If `list_file` doesn't exist → exit 2 (missing)
2. If `list_file` is empty (0 non-comment, non-blank lines) → exit 2
3. For every path in `list_file`: check `os.path.isfile(path)` **and**
   `os.path.getsize(path) > 0`
4. If ALL paths are valid files with size > 0 → print count, exit 0 (complete)
5. If ANY paths are missing or zero-byte → print valid/total counts, exit 1

**Why `isfile + size > 0`:** The downstream Python scripts define completeness
the same way (`preprocess_clean_speech.py:287-291`,
`prepare_background_music.py:403-406`). Bare `os.path.exists()` would accept
zero-byte files left by a crash mid-write.

**Performance:** `stat()` syscalls on 128K paths take ~1–2 seconds with a warm
VFS cache. Acceptable as a gate check.

### Phase 1: CHAINS Preparation

**Current behavior:** Always calls `prepare_chains_speech.py`, which has
per-file resume. Output list is written **atomically** at the end.

**Skip condition:**
```
verify_file_list "${CHAINS_LIST}" → exit 0
```

The CHAINS corpus is fixed (same source dir every run). If the list exists and
all paths are valid files with size > 0, the phase completed successfully.

The CHAINS list includes both original mono files and extracted RSI speaker
channel WAVs (`prepare_chains_speech.py:172-205`). All referenced paths must
exist for the check to pass.

**Post-skip action:** Always regenerate the merged clean list:
```bash
merge_unique_file_lists "${COMBINED_CLEAN_LIST}" "${CLEAN_LIST}" "${CHAINS_LIST}"
```

**Log on skip:**
```
[skip] CHAINS preparation: N files verified on disk (Xs)
```

### Phase 2: Clean-Speech Preprocessing

**Current behavior:** Always calls `preprocess_clean_speech.py`, which has
per-file resume and **incremental** list writes every 30 seconds.

**Skip condition:** Two-part exact check:

1. **Output list validation:** `verify_file_list "${PREPROCESS_OUTPUT_LIST}"`
   passes (all referenced files exist with size > 0).
2. **Input coverage:** For every path in the **input** list, compute its
   expected output path and verify it exists on disk with size > 0.

The expected output path transformation:
```
output_path = PREPROCESS_OUTPUT_ROOT / relpath(input_path, PREPROCESS_BASE_DIR)
             with extension changed to .wav
```

This is implemented as a dedicated Python heredoc:
```
check_preprocess_complete <input_list> <output_root> <base_dir>
```

**Why both checks are needed:**
- Check 1 alone is insufficient: the preprocess script writes the list
  incrementally (every 30s), so a crash can leave a partial list where all
  referenced files exist but not all inputs are covered.
- Check 2 catches both partial lists AND newly added input files.
- Together they provide an exact "every input has a valid output" guarantee.

**Log on skip:**
```
[skip] Clean-speech preprocessing: N/M inputs have valid outputs (Xs)
```

**Log on re-run:**
```
[check] Clean-speech preprocessing: N/M inputs covered, K pending → running phase
```

### Phase 3: Background-Music Preparation

**Current behavior:** Always calls `prepare_background_music.py`, which has
per-file resume. Output list is written **atomically** at the end.

**Skip condition:** Two-part check:

1. **Output list validation:** `verify_file_list "${MUSIC_PREPARE_OUTPUT_LIST}"`
   passes.
2. **Input coverage:** For every path in the sanitized music input list, check
   that at least variant 0's output file exists on disk with size > 0.

The expected variant-0 output path:
```
output_path = MUSIC_PREPARE_OUTPUT_ROOT
            / relpath(parent(input_path), MUSIC_PREPARE_BASE_DIR)
            / f"{stem}.variant_0.{MUSIC_PREPARE_STYLE}.wav"
```

This is implemented as a dedicated Python heredoc:
```
check_music_prep_complete <input_list> <output_root> <base_dir> <style>
```

**Why check variant 0 specifically:** If variant 0 exists for every source, all
variants were rendered (the script processes all variants per source before
moving to the next source; a crash mid-source would leave variant 0 missing
for that source).

**Post-skip action:** Always regenerate the merged music list:
```bash
merge_unique_file_lists "${MUSIC_PREPARE_MERGED_LIST}" \
    "${MUSIC_LIST_INPUT}" "${MUSIC_PREPARE_OUTPUT_LIST}"
```

**Log on skip:**
```
[skip] Background-music preparation: N sources fully prepared (Xs)
```

### Phase 4: Audio Cache Build

**Current behavior:** Always calls `build_audio_cache` with `--resume`. Has
stale-index auto-repair (rebuilds from shards when disk shard count > index
shard count).

**Decision: Always run with `--resume`. Do not add a skip check.**

Rationale:
- The cache build already has robust shard-based resume logic.
- Skipping the cache build requires validating that the index is complete AND
  trustworthy AND matches current inputs — which is complex and fragile.
- The `--resume` path is efficient: it loads the index, compares against input
  lists, and exits quickly when everything is already cached.
- The cache build is the "source of truth" consumer — it should always get a
  chance to validate its own state.

This is consistent with the user's principle: **shards are the source of
truth**, and the cache builder already knows how to read them.

**Log (unchanged):**
```
Starting audio cache build...
Resume mode is enabled - previously cached files will be skipped.
```

---

## CLI Changes

### New flag: `--force`

Disables all phase-level skip checks for phases 1–3. Every phase runs
unconditionally (existing per-file resume still applies unless `--*-overwrite`
is also set).

```bash
--force    Run all phases regardless of existing outputs
```

### Interaction with existing flags

| Flag | Phase-level skip (1–3) | Per-file resume |
|------|------------------------|-----------------|
| (default) | ✅ Skip if complete | ✅ Resume incomplete |
| `--force` | ❌ Always run | ✅ Resume incomplete |
| `--preprocess-overwrite` | ❌ Always run preprocess | ❌ Overwrite preprocess files |
| `--music-prepare-overwrite` | ❌ Always run music prep | ❌ Overwrite music files |
| `--force` + `--*-overwrite` | ❌ Always run | ❌ Overwrite all |

**Key decision:** The `--*-overwrite` flags imply `--force` for their
respective phase. If you're asking to overwrite files, you clearly want the
phase to run.

---

## Output / UX

### All phases skipped

```
==============================================
DeepFilterNet MLX Audio Cache Builder
==============================================
Profile:            apple
...

[skip] CHAINS preparation: 684 files verified on disk (0.1s)
[skip] Clean-speech preprocessing: 128,238/128,238 inputs have valid outputs (1.2s)
[skip] Background-music preparation: 2,000 sources fully prepared (0.2s)

Starting audio cache build...
Resume mode is enabled - previously cached files will be skipped.
[timing] Audio cache build: 0m08s

==============================================
Build complete!
==============================================
```

### Mixed scenario (some skipped, some run)

```
[skip] CHAINS preparation: 684 files verified on disk (0.1s)
[skip] Clean-speech preprocessing: 128,238/128,238 inputs have valid outputs (1.2s)
[check] Background-music preparation: 1,200/2,000 sources covered, 800 pending → running phase

Preparing degraded background-music variants before cache build...
...
[timing] Background-music preparation: 15m22s

Starting audio cache build...
Resume mode is enabled - previously cached files will be skipped.
```

### Force mode

```
[force] Skipping all phase completion checks (--force)

Preparing CHAINS clean-speech additions...
...
```

---

## Implementation Notes

### Where the verification code lives

All verification logic is embedded directly in `build_mlx_datastore.sh` as
inline Python heredocs (consistent with existing helpers like
`file_list_entry_stats()`, `sanitize_existing_file_list()`, etc.). No new
Python files needed.

Three functions:
1. `verify_file_list <list_file>` — generic list validator (used by phases 1, 3)
2. `check_preprocess_complete <input_list> <output_root> <base_dir>` — phase 2
3. `check_music_prep_complete <input_list> <output_root> <base_dir> <style>` — phase 3

### Error handling

- Verification function failures (Python crash, permission error) → treat as
  "phase not complete" and run the phase. Never fail the entire script because
  of a skip-check error.
- `set -euo pipefail` interaction: verification functions use explicit exit
  codes and are called in `if` conditionals, so non-zero exits don't trigger
  `set -e`.

### Performance budget

Phase skip checks should complete in < 5 seconds total for all phases. The
`os.path.isfile() + getsize()` approach achieves this easily (stat syscalls are
fast, especially with warm VFS cache).

### Stale list recovery

When a phase is NOT skipped (verification fails), the phase runs normally. The
Python script's internal resume handles per-file deduplication. At the end of a
successful run, the output list is rewritten with current, accurate data. This
naturally repairs stale lists without any special recovery logic.

### Merged list consistency

Merged lists (`clean_all.with_chains.txt`, `background_music.prepared_merged.txt`)
are always regenerated from their component lists, even when the producer phase
is skipped. This ensures the merged list reflects the current state of all
component lists, not a stale snapshot from a prior run.

---

## Scope Boundaries

**In scope:**
- Phase-level skip checks for phases 1–3 (prep phases)
- Phase 4 always runs with `--resume` (already idempotent via shards)
- `--force` CLI flag
- `--*-overwrite` implies force for that phase
- Merged list regeneration on skip
- Clear skip/re-run log messages

**Out of scope:**
- Modifying the Python scripts' internal resume logic (already works well)
- Adding sentinel/marker files (rejected in approach selection)
- Parameter change detection / fingerprinting (use `--force`; see Limitation)
- Partial-phase skipping (e.g., "skip speech shards but rebuild music shards")
- `--status` / `--dry-run` mode (nice-to-have, not part of this design)
- Pruning stale outputs when inputs are removed (additive-only model)
