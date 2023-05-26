#!/bin/bash
#SBATCH --chdir /scratch/anonymous/synth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH -o ./logs/slurm_logs/%x-%j.out
#
# Script to evaluate the checkpoints of the experiment.
#

# exit when any command fails
set -e

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate causalorca
export PYTHONPATH="$PYTHONPATH:$PWD/src"

OOD_ID=2.1

for SEED in {72001..72050}; do
  python -m causalorca.bin.generate_scenes \
    --layout mixed_scenario_1 \
    --num_scenes 100 \
    --curvature_cutoff 0.25 \
    --min_num_ped 14 \
    --max_num_ped 18 \
    --logs_path ./logs/synth_v1.a.odd.${OOD_ID}_${SEED}__$(date +"%F_%T.%N")/ \
    --seed $SEED \
    --scene_must_have_nc_ego &
done

FAIL=0
for job in $(jobs -p); do
  echo "$job"
  wait "$job" || let "FAIL+=1"
done
echo "$FAIL"
if [ "$FAIL" == "0" ]; then
  echo "All bin.generate_scenes subprocesses finished successfully."
else
  echo "${FAIL} bin.generate_scenes subprocesses DID NOT finish successfully."
fi

echo "Merging intermediary results..."
python -m tools.merge_generated_datasets \
  --dataset_paths logs/synth_v1.a.odd.${OOD_ID}_*/*.pkl \
  --output_merged_dataset_path "data/synth_v1.a.odd.${OOD_ID}.test.300.pkl" \
  --scene_number_limit 300
python -m tools.merge_generated_datasets \
  --dataset_paths logs/synth_v1.a.odd.${OOD_ID}_*/*.pkl \
  --output_merged_dataset_path "data/synth_v1.a.odd.${OOD_ID}.test.pkl"
echo "Done."
