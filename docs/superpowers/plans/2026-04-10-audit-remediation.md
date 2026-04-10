# Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 13 remaining issues from the 2026-04-10 adversarial review of the dataset pipeline.

**Architecture:** Surgical edits to two bash scripts (`download_datasets.sh`, `build_mlx_datastore.sh`), one test file, and the audit report to mark issues as FIXED. Changes are independent per-issue, grouped into logical tasks.

**Tech Stack:** Bash, Python (inline heredocs), pytest

---

## File Map

| File | Changes |
|------|---------|
| `scripts/datasets/download_datasets.sh` | Tasks 1, 2, 3, 5, 6, 7 |
| `scripts/datasets/build_mlx_datastore.sh` | Tasks 4, 8, 9 |
| `DeepFilterNet/tests/test_dataset_scripts.py` | Task 3 |
| `docs/audits/2026-04-10-adversarial-review.md` | Task 10 (update statuses) |

---

### Task 1: Fix Zipfile Path Traversal in Download Script

**Files:**
- Modify: `scripts/datasets/download_datasets.sh:1195-1203`

- [ ] **Step 1: Add path normalization to zip extraction**

Replace the Python zip extraction snippet (lines 1195-1203) with a safe version that rejects `../` paths:

```python
python3 -c "
import zipfile, sys, os
z = zipfile.ZipFile(sys.argv[1])
dest = os.path.realpath(sys.argv[2])
for info in z.infolist():
    target = os.path.realpath(os.path.join(dest, info.filename))
    if not target.startswith(dest + os.sep) and target != dest:
        print(f'SECURITY: skipping path traversal entry: {info.filename}', file=sys.stderr)
        continue
    if not os.path.exists(target):
        z.extract(info, dest)
" "${archive}" "${stage_dir}"
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n scripts/datasets/download_datasets.sh`
Expected: No errors

---

### Task 2: Secure Aria2 Input File Permissions

**Files:**
- Modify: `scripts/datasets/download_datasets.sh:972`

- [ ] **Step 1: Add chmod after file creation**

After line 972 (`: > "${ARIA2_INPUT_FILE}"`), add:

```bash
  chmod 600 "${ARIA2_INPUT_FILE}"
```

This restricts the file before any auth tokens are appended.

- [ ] **Step 2: Verify syntax**

Run: `bash -n scripts/datasets/download_datasets.sh`
Expected: No errors

---

### Task 3: Fix Failing Production Profile Test

**Files:**
- Modify: `DeepFilterNet/tests/test_dataset_scripts.py:1925`

- [ ] **Step 1: Add --no-download-mtg-jamendo to test args**

The test uses `--profile production` but doesn't disable MTG-Jamendo. Since production
enables MTG-Jamendo by default, the fake aria2 stub receives the MTG-Jamendo download
queue and fails with exit 99. Add `"--no-download-mtg-jamendo",` after the existing
`"--no-download-acousticrooms",` line.

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python3 -m pytest DeepFilterNet/tests/test_dataset_scripts.py::test_download_datasets_zenodo_range_download_bypasses_aria2_and_extracts_vctk -v`
Expected: PASS

---

### Task 4: Rename MAX_PENDING_BYTES to MAX_PENDING_GB

**Files:**
- Modify: `scripts/datasets/build_mlx_datastore.sh` (lines 553-554, 745, 831, 1114)

- [ ] **Step 1: Rename variable throughout**

1. Parser (line 554): `CLI_MAX_PENDING_BYTES="$2"` → `CLI_MAX_PENDING_GB="$2"`
2. Variable assignment (line 745): `MAX_PENDING_BYTES=...` → `MAX_PENDING_GB=...` (keep `${MAX_PENDING_BYTES:-8}` as env-var fallback for compat, then `${MAX_PENDING_GB:-8}`)
3. Config banner (line 831): `${MAX_PENDING_BYTES}` → `${MAX_PENDING_GB}`
4. Python call (line 1114): `--max-pending-bytes "${MAX_PENDING_BYTES}"` → `--max-pending-bytes "${MAX_PENDING_GB}"`

Note: the Python flag name stays `--max-pending-bytes` because the Python side does the
GB→bytes conversion internally. Only the bash variable name is misleading.

- [ ] **Step 2: Add default to help text**

Change help text (line 340) from:
```
  --max-pending-gb N          Max in-flight async shard writer budget in GB
```
to:
```
  --max-pending-gb N          Max in-flight async shard writer budget in GB (default: 8)
```

- [ ] **Step 3: Verify syntax**

Run: `bash -n scripts/datasets/build_mlx_datastore.sh`
Expected: No errors

---

### Task 5: Fix should_download() Dead Code

**Files:**
- Modify: `scripts/datasets/download_datasets.sh:954`

- [ ] **Step 1: Change catch-all to return 1**

Line 954: `apple|prototype|*) return 0` → `apple|prototype|*) return 1`

This makes the default "don't download" for unknown/empty flags, which is the safe
default. The actual profile-specific defaults (lines 1591-1620) set flags to "1" or "0"
before should_download() is called, so this change only affects the unreachable edge case.

- [ ] **Step 2: Verify syntax**

Run: `bash -n scripts/datasets/download_datasets.sh`
Expected: No errors

---

### Task 6: Fix Aria2 Queue Re-download Verification Gap

**Files:**
- Modify: `scripts/datasets/download_datasets.sh:1444`

- [ ] **Step 1: Guard second verify_archive call**

Replace lines 1443-1446:
```bash
      download_file "${url}" "${archive}"
      verify_archive "${archive}"
    fi
    extract_archive "${archive}" "${dest}"
```
with:
```bash
      download_file "${url}" "${archive}"
      if ! verify_archive "${archive}"; then
        echo "[error] archive still fails verification after re-download, skipping: ${archive}" >&2
        continue
      fi
    fi
    extract_archive "${archive}" "${dest}"
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n scripts/datasets/download_datasets.sh`
Expected: No errors

---

### Task 7: Add Error Handling to Zip Extraction + Use Checksum in Cache Lookup

**Files:**
- Modify: `scripts/datasets/download_datasets.sh:919-921`

- [ ] **Step 1: Add checksum to cache lookup**

Replace lines 919-921 to match on path, size, AND checksum:
```bash
  local checksum
  checksum="$(checksum_file "${path}")"
  awk -F'\t' -v p="${path}" -v s="${size}" -v c="${checksum}" \
    '$1==p && $2==s && $4==c {found=1} END {exit(found?0:1)}' \
    "${VERIFY_CACHE_FILE}" >/dev/null 2>&1
```

Update the comment on line 917 accordingly.

- [ ] **Step 2: Verify syntax**

Run: `bash -n scripts/datasets/download_datasets.sh`
Expected: No errors

---

### Task 8: Add Default Value to --min-duration Help Text

**Files:**
- Modify: `scripts/datasets/build_mlx_datastore.sh:337`

- [ ] **Step 1: Update help text**

Change:
```
  --min-duration SEC          Minimum clean-speech duration before skip/merge
```
to:
```
  --min-duration SEC          Minimum clean-speech duration before skip/merge
                              (default: same as --segment-length, typically 5.0)
```

---

### Task 9: Improve Stderr Handling in Verification Functions

**Files:**
- Modify: `scripts/datasets/build_mlx_datastore.sh` (lines 900, 945, 1035)

- [ ] **Step 1: Replace blind 2>/dev/null with conditional stderr**

For each verification call, redirect stderr to a temp file and show it only on unexpected
failure (not on expected "phase incomplete" exits). Update the three call sites:

Phase 1 (CHAINS, line 900):
```bash
    if verify_result="$(verify_file_list "${CHAINS_LIST}" 2>"${TMPDIR:-/tmp}/.dfn_verify_chains.err")"; then
```
Then after the if/fi, if the phase is running (not skipped), print the stderr:
```bash
    [[ -s "${TMPDIR:-/tmp}/.dfn_verify_chains.err" ]] && cat "${TMPDIR:-/tmp}/.dfn_verify_chains.err" >&2
```

Apply the same pattern to phases 2 and 3.

---

### Task 10: Run Full Test Suite, Commit, and Update Audit Report

**Files:**
- Modify: `docs/audits/2026-04-10-adversarial-review.md`

- [ ] **Step 1: Run all dataset tests**

Run: `.venv/bin/python3 -m pytest DeepFilterNet/tests/test_dataset_scripts.py -v`
Expected: All tests pass

- [ ] **Step 2: Run bash syntax checks**

Run: `bash -n scripts/datasets/download_datasets.sh && bash -n scripts/datasets/build_mlx_datastore.sh`
Expected: No errors

- [ ] **Step 3: Update audit report statuses**

Change status of all fixed findings from "VERIFIED" to "FIXED" in the audit report.

- [ ] **Step 4: Commit and push**

```bash
git add -A
git commit -m "fix(datasets): remediate all adversarial review findings

- Zip extraction: add path traversal protection (realpath check)
- Aria2: chmod 600 input file before writing auth tokens
- Test: add --no-download-mtg-jamendo to production profile test
- Rename MAX_PENDING_BYTES→MAX_PENDING_GB for clarity
- should_download(): change catch-all default to return 1 (safe)
- Extract queue: guard second verify_archive with if/continue
- Cache lookup: include checksum in cache match (path+size+sha256)
- Help text: add default values for --min-duration, --max-pending-gb
- Verification stderr: capture to temp file, show on unexpected failures"
git push
```
