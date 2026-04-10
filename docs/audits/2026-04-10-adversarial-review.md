# Adversarial Codebase Review — Dataset Pipeline

**Date:** 2026-04-10  
**Scope:** `scripts/datasets/` pipeline — download, preparation, and MLX cache build  
**Branch:** `feat/background-music-dataset`  
**Commit:** `be28d61`

---

## Executive Summary

The dataset pipeline is **functionally operational** — downloads, preparation phases, and cache
building all work end-to-end. The idempotency mechanism (Phase 1-3 skip logic) is correctly
implemented with one latent bug in the `_external` path fallback. The download script has
known security weaknesses typical of research-grade infrastructure but no show-stoppers
for intended use (trusted datasets from known sources).

**Blocking issues:** 1  
**Important issues:** 5  
**Improvements:** 8  

---

## Findings

### 🔴 CRITICAL — `_external` Path Fallback Missing in Verification Functions

**Status:** VERIFIED  
**Evidence:** `build_mlx_datastore.sh` lines 242-244 (`check_preprocess_complete`) and
lines 287-288 (`check_music_prep_complete`) — both `continue` on `ValueError` from
`relative_to()`, while the Python scripts they verify use `Path("_external") / source.name`
as fallback (`preprocess_clean_speech.py:97-98`, `prepare_background_music.py:192-193`).

**Impact:** If any input files are outside `base_dir`, the verification functions silently
skip them, potentially claiming "100% covered" when external files weren't checked. This
could cause the phase to be incorrectly skipped on re-run, leaving external files unprocessed.

**Mitigating factor:** Verified that zero of the 128,238 entries in the current
`clean_all.with_chains.txt` are external to the adjusted base dir. The bug is latent.

**Fix:** Add `_external` fallback in both functions:
```python
except ValueError:
    rel = Path("_external") / inp.name
```

---

### 🟡 HIGH — Test Failure in Download Script Test Suite

**Status:** VERIFIED  
**Evidence:** `pytest DeepFilterNet/tests/test_dataset_scripts.py` — 1 failed, 18 passed:
```
FAILED test_download_datasets_zenodo_range_download_bypasses_aria2_and_extracts_vctk
assert 99 == 0  # aria2c exit 99
```
The test spins up a local HTTP server for VCTK but the production profile also tries to
download FSD50K via aria2, which fails against the local server.

**Impact:** Test suite does not pass cleanly — the production-profile integration test is
broken.

**Fix:** The test needs to either mock all production-profile downloads or use a more
targeted profile. Pre-existing issue, not caused by idempotency changes.

---

### 🟡 HIGH — Zipfile Path Traversal in Archive Extraction

**Status:** VERIFIED  
**Evidence:** `download_datasets.sh` line 1200:
```python
out = os.path.join(dest, info.filename)
```
No `../` stripping or path normalization. On Python < 3.12 this allows zip entries with
`../../../etc/foo` to escape the extraction directory.

**Impact:** Malicious zip archives from compromised download sources could write arbitrary
files. Mitigated by: (a) all download URLs are hardcoded to known-good sources, (b) Python
3.12+ has `zipfile` protections.

**Fix:** Add `os.path.commonpath([dest, os.path.realpath(out)])` check or use
`shutil.unpack_archive()`.

---

### 🟡 HIGH — `--max-pending-bytes` Flag Name Misleading

**Status:** VERIFIED  
**Evidence:** CLI flag is `--max-pending-gb` (help text says "GB", line 341), stored into
`MAX_PENDING_BYTES` variable, passed to Python as `--max-pending-bytes`. Python's argparse
names the dest `max_pending_bytes` (line 888) but the help says "GB" (line 891) and
conversion happens at line 1053: `args.max_pending_bytes * 1024 * 1024 * 1024`.

**Impact:** No functional bug (the conversion is correct), but the inconsistent naming
across bash variable, Python argparse dest, and actual semantics (GB not bytes) is a
maintenance trap. A developer reading the bash code would assume bytes.

**Fix:** Rename to `--max-pending-gb` end-to-end in both bash and Python, or document
the units clearly in variable names (`MAX_PENDING_GB`).

---

### 🟡 HIGH — GitHub Token in Aria2 Input File Without Restricted Permissions

**Status:** VERIFIED  
**Evidence:** `download_datasets.sh` line 1365:
```bash
echo "  header=Authorization: token ${gh_token}"
```
Written to `${ARIA2_INPUT_FILE}` which is created with default umask permissions.

**Impact:** On shared systems, the GitHub auth token is readable by other users. The token
is ephemeral (from `gh auth token`) and read-only, but exposure is still undesirable.

**Fix:** `chmod 600 "${ARIA2_INPUT_FILE}"` before writing sensitive content.

---

### 🟡 HIGH — Python Script Arguments Not Exposed via Bash

**Status:** VERIFIED  
**Evidence:** Cross-reference of bash invocations vs Python argparse:

| Script | Unexposed Argument | Default | Impact |
|--------|-------------------|---------|--------|
| `prepare_chains_speech.py` | `--overwrite` | off | Can't force CHAINS rebuild |
| `prepare_chains_speech.py` | `--num-workers` | cpu_count | Can't tune parallelism |
| `prepare_background_music.py` | `--rir-probability` | 0.8 | Can't tune RIR application rate |
| `build_audio_cache.py` | `--rebuild-index` | off | Can't force index rebuild |
| `build_audio_cache.py` | `--p-clipping` | 0.0 | Can't enable clipping augmentation |

**Impact:** Users can't control these parameters through the bash orchestrator. The defaults
are sensible but power users may want control.

**Fix:** Add corresponding `--chains-*`, `--music-prepare-rir-probability`, etc. flags as
needed. Low priority since defaults are reasonable.

---

### 🟠 MEDIUM — `should_download()` Has Unreachable Default Branch

**Status:** VERIFIED  
**Evidence:** Lines 949-957 — the `case` block always returns 0 for every profile.
However, this code is unreachable because all `DOWNLOAD_*` variables are explicitly set to
"0" or "1" at lines 1591-1620 (profile-specific defaults), so the `flag` parameter is
never empty when `should_download()` is called.

**Impact:** Dead code. If profile defaults were ever removed, all datasets would download
regardless of profile (the catch-all `*` returns 0). Should differentiate profiles.

**Fix:** Make the catch-all case return 1 (default off) so undeclared datasets don't
auto-download:
```bash
apple|prototype|*) return 1 ;;
```

---

### 🟠 MEDIUM — Stderr Suppression in Verification Functions

**Status:** VERIFIED  
**Evidence:** Lines 901, 945, 1035 — all verification calls use `2>/dev/null`, hiding
Python tracebacks, import errors, and permission failures.

**Impact:** When verification fails due to a real error (not just "incomplete"), debugging
is harder because the error message is suppressed. The fallback (run the phase) is correct
but the user gets no indication of why the check failed.

**Fix:** Log stderr to a temp file and display on unexpected failures:
```bash
if ! result="$(check_preprocess_complete ... 2>"${tmpfile}")"; then
  [[ -s "${tmpfile}" ]] && cat "${tmpfile}" >&2
fi
```

---

### 🟠 MEDIUM — Preprocess Base Dir Not Set Before Skip Check

**Status:** VERIFIED  
**Evidence:** Line 937 — `PREPROCESS_BASE_DIR_TO_USE` is computed inside the
`PREPROCESS_CLEAN_SPEECH` conditional, then used in the skip check at line 945. This is
correct in the current code. However, if `INCLUDE_CHAINS` is enabled, the `compute_common_base_dir`
call requires the merged list (`CLEAN_LIST_TO_USE`) to already exist, which it does because
Phase 1 always regenerates the merged list.

**Impact:** No current bug, but the dependency chain is fragile — if Phase 1 merged list
regeneration were ever moved inside the skip conditional, Phase 2's skip check would break.

**Fix:** Document the dependency in a comment.

---

### 🟠 MEDIUM — No Disk Space Check Before Multi-Hour Operations

**Status:** VERIFIED  
**Evidence:** Neither `download_datasets.sh` nor `build_mlx_datastore.sh` checks available
disk space before starting. The production profile downloads ~200 GB and the cache build
can produce ~50+ GB.

**Impact:** Out-of-space failures mid-operation produce confusing errors. The idempotency
mechanism handles this well (resume from where it stopped), but a pre-flight check would
save hours of wasted work.

**Fix:** Add `df -h "${OUTPUT_DIR}"` display in config banner and optional `--min-free-gb`
guard.

---

### ⚪ LOW — `stat_mtime()` Return Value Stored But Never Used for Cache Lookup

**Status:** VERIFIED  
**Evidence:** `download_datasets.sh` line 730 — `stat_mtime()` defined and used in cache
storage (line 928), but cache lookup at lines 919-920 matches only on path and size, not
mtime.

**Impact:** The mtime column in the verification cache is wasted. If an archive is
re-downloaded to the exact same size but different content, the stale cache entry would
match.

**Fix:** Either remove mtime from cache or add it to the lookup key.

---

### ⚪ LOW — Config Banner Missing Default Values for Some Options

**Status:** VERIFIED  
**Evidence:** Help text for `--min-duration` (line 339) and `--max-pending-gb` (line 341)
don't show their default values. Other options do (e.g., `--shard-size` says "default: 500").

**Impact:** User confusion about what happens when the flag is omitted.

**Fix:** Add `(default: SEGMENT_LENGTH)` and `(default: 8)` to help text.

---

### ⚪ LOW — Archive Extraction Error Handling Inconsistent

**Status:** VERIFIED  
**Evidence:** `download_datasets.sh` — tar extraction (lines 1187-1190) relies on
`set -euo pipefail` but zip extraction (lines 1195-1203) uses inline Python without
explicit error handling. Both are in `case` branches inside a function.

**Impact:** Partial extraction may go undetected for zip archives specifically. Tar
extraction fails fast due to `set -e`.

**Fix:** Add `try/except` with `sys.exit(1)` in the Python zip extraction snippet.

---

### ⚪ LOW — Aria2 Queue Verification Gap After Re-Download

**Status:** SUSPECTED (not runtime-verified)  
**Evidence:** `download_datasets.sh` line 1444 — after a re-download,
`verify_archive "${archive}"` runs standalone (not inside `if ! ...`), meaning a second
verification failure doesn't prevent extraction.

**Impact:** A persistently corrupted archive would be re-downloaded and then extracted
without verification passing. The `set -e` trap should catch a non-zero exit from
`verify_archive`, but the error path is unclear.

**Fix:** Wrap in explicit `if ! verify_archive ...; then continue; fi`.

---

## Coverage Matrix

| Review Surface | Status | Notes |
|---------------|--------|-------|
| `build_mlx_datastore.sh` | ✅ Reviewed | Full structural + runtime verification |
| `download_datasets.sh` | ✅ Reviewed | Structural analysis + partial runtime (tests) |
| `prepare_background_music.py` | ✅ Reviewed | Interface cross-ref + path logic verified |
| `preprocess_clean_speech.py` | ✅ Reviewed | Interface cross-ref + path logic verified |
| `prepare_chains_speech.py` | ✅ Reviewed | Interface cross-ref |
| `build_audio_cache.py` | ✅ Reviewed | Interface cross-ref + runtime (cache build) |
| `curate_background_music.py` | ⚠️ Partial | Not structurally audited; called by download script |
| `fsd50k_filter.py` | ⚠️ Partial | Not structurally audited |
| `bsdtar_header.py` | ⚠️ Partial | Security surface noted but not deep-audited |
| `zip_merge_progress.py` | ⚠️ Partial | Security surface noted |
| `audition_background_music.py` | ❌ Skipped | Display-only utility, low risk |
| `audb_download.py` | ⚠️ Partial | Noted dependency installation risk |
| `build_hdf5.sh` | ❌ Skipped | Legacy path, not part of active MLX pipeline |
| Test suite | ✅ Reviewed | 18 passed, 1 failed (pre-existing) |
| CI integration | ✅ Reviewed | No dataset scripts in CI (noted as gap) |
| Idempotency design spec | ✅ Reviewed | Cross-referenced against implementation |
| Documentation (DATASETS.md) | ✅ Reviewed | Claims verified against code |

---

## Remediation Priorities

### Blocking (fix before next release)
1. **`_external` path fallback in verification functions** — latent bug that will cause
   incorrect phase skipping if any user has files outside `base_dir`

### Important (fix soon)
2. **Failing test** — `test_download_datasets_zenodo_range_download_bypasses_aria2_and_extracts_vctk`
   needs to be fixed or marked as expected-failure for production profile
3. **Zipfile path traversal** — add path normalization to zip extraction
4. **Aria2 token file permissions** — `chmod 600` before writing auth headers
5. **`should_download()` catch-all** — change default branch to return 1

### Improvements (lower urgency)
6. Expose missing Python arguments via bash (especially `--rir-probability`)
7. Reduce stderr suppression in verification functions
8. Add disk space pre-flight check
9. Rename `MAX_PENDING_BYTES` to `MAX_PENDING_GB` for clarity
10. Add default values to help text for `--min-duration` and `--max-pending-gb`
11. Add explicit error handling to inline Python zip extraction
12. Fix aria2 queue re-download verification gap
13. Remove or use `stat_mtime()` in cache lookup
