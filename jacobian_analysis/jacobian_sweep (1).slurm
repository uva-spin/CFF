#!/bin/bash
# =============================================================================
# jacobian_sweep.slurm
#
# Slurm array job: run jacobian_worker.py for each kinematic point in
# kin_grid.json.  Each task is independent.
#
# HOW TO USE:
#   1.  Run gen_kin_grid.py     # creates kin_grid.json and prints the number kinematic points
#   2.  Edit --array=0-<N-1> below
#   3.  sbatch jacobian_sweep.slurm
#   4.  python aggregate_results.py     # (after all tasks finish)
#
# RESOURCE GUIDANCE:
#   - Each task runs one kinematic point (31 subset Jacobians).
#     With PHI_N=50, each task finishes in < 30 seconds.
#   - Request 1 cpu per task; use array parallelism for throughput.
#   - Standard partition is fine; no GPU needed.
#   - Memory: < 500 MB per task.
#
# BEFORE SUBMITTING:
#   - Set --array range to match N in kin_grid.json  (currently 0-53)
#   - Set WORKDIR to the absolute path of this directory
# =============================================================================

#SBATCH --job-name=jac_sweep
#SBATCH --partition=standard
#SBATCH --account=spinquest_standard      # allocation
#SBATCH --array=0-53                   # Should be 0 to (N_points - 1)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-00:10:00             # 10 minutes is generous for each task
#SBATCH --output=logs/task_%A_%a.out  # %A=job ID, %a=array index
#SBATCH --error=logs/task_%A_%a.err

# ---- Environment setup ----
# I have to do this because the default Rivanna python is too old
module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

# ---- Paths ----
# Set to the directory containing this script, jacobian_worker.py,
# cross_section_script.py, and kin_grid.json.
WORKDIR="/scratch/mbg8jy/jacobian_sweep"    # <-- EDIT: your working directory
cd "$WORKDIR" || { echo "Cannot cd to $WORKDIR"; exit 1; }

# Create output dirs if they don't exist
mkdir -p logs results

# ---- Run the worker ----
echo "=== Task $SLURM_ARRAY_TASK_ID starting on $(hostname) at $(date) ==="

python3 jacobian_worker.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --grid    "$WORKDIR/kin_grid.json" \
    --outdir  "$WORKDIR/results" \
    --k-beam  5.75 \
    --phi-n   50 \
    --fd-eps  1e-3 \
    --rank-tol 1e-3

EXIT_CODE=$?
echo "=== Task $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE at $(date) ==="
exit $EXIT_CODE
