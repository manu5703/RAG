#!/bin/bash
# Full answer eval + hallucination detector on a CHPC GPU node.
#   sbatch scripts/chpc_answer_eval.slurm
# Find your account and partitions with `mychpc batch`. notchpeak-shared-short is
# non-preemptible but capped at 8 cores and 8 hours, which is plenty for this.
#SBATCH --job-name=campus-rag-eval
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
module load miniforge3 2>/dev/null || true
source activate campus-rag 2>/dev/null || true   # conda env with requirements.txt installed

# keep model weights out of the 50 GB home quota
export HF_HOME=/scratch/general/vast/$USER/hf-cache

python -m eval.retrieval_eval
python -m eval.answer_eval --modes bm25 dense hybrid hybrid_rerank --lens
python -m interpretability.detector eval/results/answers_hybrid_rerank.jsonl
