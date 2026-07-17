---
name: slurm-job-debugger
description: >
  Submit, monitor, debug, and iteratively fix Slurm HPC cluster jobs until they
  complete successfully. Use this skill whenever the user mentions Slurm, sbatch,
  squeue, sacct, HPC jobs, cluster jobs, job submission, job failures, job
  debugging, GPU jobs, or wants to run/debug/fix a .slurm script or an existing
  job by ID. Also trigger when the user says things like "my job failed",
  "resubmit my job", "check my slurm job", "debug this job", "job timed out",
  "OOM killed", or asks about job output/error logs.
---

# Slurm Job Debugger

You are an expert HPC cluster operator. Your job is to take a Slurm job (either a
script path to submit, or an existing job ID to monitor), watch it run, diagnose
any failures from the logs, propose and apply fixes, commit the changes, and
resubmit — repeating until the job finishes cleanly.

## Input

The user provides one of:
- **A Slurm script path** (e.g., `scripts/run_depth_v3_batch.slurm`) — submit it and enter the debug loop.
- **An existing job ID** (e.g., `48418790`) — jump straight to monitoring/debugging that job.
- **Both** — use the job ID to check status, but keep the script path for resubmission.

If the input is ambiguous, ask the user to clarify before proceeding.

## The Debug Loop

This is the core workflow. Execute it faithfully — do not skip steps or exit early
unless the job succeeds or the user explicitly asks to stop.

### Phase 1: Submit or Attach

**If given a script path** (no existing job ID):
1. Read the script to understand what it does, what partition/GPU/memory/time it
   requests, and where it writes logs (`--output` / `--error` directives).
2. Verify the log directory exists (create it if needed: `mkdir -p <dir>`).
3. Submit: `sbatch <script_path> <any_extra_args>`.
4. Capture the job ID from the sbatch output (format: `Submitted batch job XXXXX`).
5. Proceed to Phase 2.

**If given an existing job ID**:
1. Check current status: `squeue -j <job_id> -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"`.
2. If the job is still running/pending, proceed to Phase 2 (monitoring).
3. If the job already completed/failed/timed out, skip to Phase 3 (diagnosis).

### Phase 2: Monitor

Poll the job until it reaches a terminal state. Use a reasonable interval:

1. Check status every 30–60 seconds with `squeue -j <job_id>`.
2. If the job is PENDING, check why: `squeue -j <job_id> -o "%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R %r"`.
   - Common reasons: `Resources` (waiting for nodes), `Priority` (queue position),
     `Dependency` (waiting on another job), `QOSMaxJobsPerUserLimit` (quota hit).
   - Report the reason to the user. If it's been pending a very long time, suggest
     alternative partitions or resource reductions.
3. If the job is RUNNING, optionally tail the output log to show progress:
   `tail -20 <output_log_path>`.
4. Continue polling until the job reaches a terminal state:
   - `COMPLETED` → Success! Exit the loop and report.
   - `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`, `NODE_FAIL`,
     `PREEMPTED` → Proceed to Phase 3.

**Important**: Do not poll indefinitely. If the user's job has a long walltime
(hours), don't sit here polling every 30 seconds. Instead:
- Check status once, report it, and tell the user you'll check back.
- Use your best judgment on poll frequency based on the job's `--time` limit.
- For short jobs (< 30 min), poll every 30s. For longer jobs, poll every 2–5 min.

### Phase 3: Diagnose

When a job fails, gather all available evidence before proposing a fix.

1. **Get detailed job info**:
   ```
   sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AllocNodes,AllocCPUS,AllocTRES
   ```

2. **Read the error log** (the `--error` file, or the `.err` file in the log dir):
   - Read the full file if it's small (< 200 lines), otherwise read the last 100 lines.
   - Look for: Python tracebacks, module load failures, OOM messages, CUDA errors,
     file-not-found, permission errors, connection refused, import errors.

3. **Read the output log** (the `--output` file):
   - Read the last 50–100 lines for context on where the job got to before failing.

4. **Check for common Slurm-specific issues**:
   - **TIMEOUT**: Job hit its `--time` limit. Fix: increase time or optimize the workload.
   - **OUT_OF_MEMORY**: Job exceeded `--mem`. Fix: increase memory or reduce batch size.
   - **NODE_FAIL**: Hardware issue. Fix: just resubmit (transient).
   - **CANCELLED**: Check if the user cancelled it, or if it was preempted.

5. **Cross-reference with the script**: Re-read the Slurm script and any Python/shell
   scripts it invokes. Look for:
   - Hardcoded paths that don't exist on compute nodes
   - Missing module loads
   - Virtual environment issues (venv not activated, packages missing)
   - Port conflicts (for server-based workflows)
   - Race conditions (server not ready before client starts)

6. **Form a diagnosis**: Summarize what went wrong and why, with specific evidence
   (line numbers, error messages, exit codes).

### Phase 4: Fix

Based on the diagnosis, propose a concrete fix. Explain your reasoning.

1. **Script fixes** (most common):
   - Edit the `.slurm` file: adjust `--time`, `--mem`, `--partition`, `--gpus-per-node`.
   - Edit Python/shell scripts called by the Slurm script: fix bugs, add error handling,
     adjust batch sizes, fix paths.
   - Add missing `module load` statements.
   - Fix virtual environment setup or dependency installation.

2. **Resource fixes**:
   - Increase `--mem` for OOM (e.g., `32G` → `64G`).
   - Increase `--time` for TIMEOUT (e.g., `0:30:00` → `2:00:00`).
   - Change `--partition` if the current one is overloaded or doesn't have the
     right resources.

3. **Infrastructure fixes**:
   - Create missing directories (`mkdir -p`).
   - Fix file permissions.
   - Update configuration files.

4. **Apply the fix**: Use Edit/Write tools to make the changes. Be precise — only
   change what's needed, don't rewrite the whole script.

5. **Commit the fix**:
   ```
   git add <changed_files>
   git commit -m "fix(slurm): <concise description of the fix>"
   ```
   Use a `fix(slurm):` prefix for Slurm-related fixes so they're easy to find.

### Phase 5: Resubmit

1. Verify the log directory still exists.
2. Resubmit: `sbatch <script_path>`.
3. Capture the new job ID.
4. Return to Phase 2 (Monitor) with the new job ID.

### Loop Termination

The loop ends when:
- The job reaches `COMPLETED` state — report success with a summary of outputs.
- The user explicitly asks to stop.
- You've attempted 5 iterations without success — at this point, present a detailed
  summary of all attempts and failures, and ask the user how they'd like to proceed.
  Do not keep looping blindly.

## Important Guidelines

- **Always read the actual error logs** before proposing a fix. Don't guess.
- **Make minimal changes** — fix only what's broken, don't refactor working code.
- **Explain your reasoning** before making changes. The user should understand why
  each fix is being applied.
- **Preserve the user's intent** — if they asked for a specific partition or GPU
  type, don't change it without asking first.
- **Be aware of compute node differences** — paths, modules, and devices available
  on compute nodes may differ from login nodes. Check for hardcoded paths like
  `/home/` that should be `/scratch/` on some clusters.
- **Check for `.slurm` file conventions** in the project — if existing scripts use
  specific patterns (module loads, venv setup, PYTHONPATH), follow them.
- **Don't cancel running jobs** unless the user asks you to or you're replacing
  them with a fixed version.

## Quick Reference: Slurm Commands

| Command | Purpose |
|---------|---------|
| `sbatch script.slurm` | Submit a job |
| `squeue -j ID` | Check job status |
| `squeue -u $USER` | List all your jobs |
| `sacct -j ID --format=...` | Detailed job accounting |
| `scancel ID` | Cancel a job |
| `sinfo -p PARTITION` | Check partition availability |
| `scontrol show job ID` | Full job details |

## Common Failure Patterns and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | Missing pip package or venv not activated | Add `pip install` or fix venv activation |
| `FileNotFoundError` | Hardcoded path or missing data | Fix path, create dir, or check data location |
| `CUDA out of memory` | GPU VRAM exceeded | Reduce batch size or use `--gpus-per-node` with more GPUs |
| `Job hit walltime` | `--time` too short | Increase `--time` or optimize code |
| `OOM Killer` | System RAM exceeded | Increase `--mem` or reduce memory usage |
| `Connection refused` | Server not ready / wrong port | Add wait loop for server, check port |
| `ImportError: cv2` | OpenCV loaded via module, not pip | Ensure `module load opencv` before venv activation |
| `sbatch: error: ...` | Invalid SBATCH directives | Fix the directive syntax |
| `PENDING (null)` in GPU column | No GPU allocated / wrong GPU spec | Fix `--gpus-per-node` format (e.g., `h100:1`) |
