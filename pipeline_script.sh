#!/bin/bash
BASE_DIR=$(pwd)
DEMO_DIR="$BASE_DIR/demonstration_collection"
GMM_DIR="$BASE_DIR/gmm-gmr"

read -p "Enter the number of demonstrations you would like to collect: " num_demos

echo "Running collect_demonstration.py $num_demos times..."
cd "$DEMO_DIR" || { echo "Failed to change directory to $DEMO_DIR"; exit 1; }
for ((i=1; i<=num_demos; i++)); do
    echo "Run $i: Executing collect_demonstration.py..."
    python collect_demonstration.py || { echo "collect_demonstration.py failed on run $i"; exit 1; }
done

echo "Running graph_all_demonstrations.py..."
python graph_all_demonstrations.py || { echo "graph_all_demonstrations.py failed"; exit 1; }

echo "Running smooth_demonstrations.py..."
python smooth_demonstrations.py || { echo "smooth_demonstrations.py failed"; exit 1; }

echo "Running graph_all_smoothed_demonstrations.py..."
python graph_all_smoothed_demonstrations.py || { echo "graph_all_smoothed_demonstrations.py failed"; exit 1; }

echo "Switching to GMM-GMR directory..."
cd ..
cd "$GMM_DIR" || { echo "Failed to change directory to $GMM_DIR"; exit 1; }

echo "Running main.py..."
python main.py || { echo "main.py failed"; exit 1; }

echo "Running apply_skill.py..."
python apply_skill_to_block.py || { echo "apply_skill.py failed"; exit 1; }

echo "Pipeline execution complete."
