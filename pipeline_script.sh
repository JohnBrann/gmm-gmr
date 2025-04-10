#!/bin/bash
#BASE_DIR="/home/johnrobot/robot_learning" # modify in future to make module
BASE_DIR="."
DEMO_DIR="$BASE_DIR/demonstration_collection"
GMM_DIR="$BASE_DIR/gmm-gmr"

# --- Collect 4 demonstrations ---
echo "Running collect_demonstration.py 4 times..."
cd "$DEMO_DIR" || { echo "Failed to change directory to $DEMO_DIR"; exit 1; }
for i in {1..4}; do
    echo "Run $i: Executing collect_demonstration.py..."
    python collect_demonstration.py || { echo "collect_demonstration.py failed on run $i"; exit 1; }
done

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
python apply_skill.py || { echo "apply_skill.py failed"; exit 1; }

echo "Pipeline execution complete."
