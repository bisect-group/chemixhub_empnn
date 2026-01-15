#!/bin/bash
# Script to initialize and run wandb sweep for hyperparameter optimization

# Check if project name is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_wandb_sweep.sh <wandb_project_name> [num_agents]"
    echo "Example: ./run_wandb_sweep.sh chemixhub_roshan 4"
    exit 1
fi

PROJECT_NAME=$1
NUM_AGENTS=${2:-1}  # Default to 1 agent if not specified

# Initialize the sweep and capture the sweep ID
echo "Initializing wandb sweep..."
SWEEP_ID=$(wandb sweep --project "$PROJECT_NAME" ../config/wandb_sweep_reaxo.yaml 2>&1 | grep -oP 'wandb agent .* \K[a-z0-9/]+$')

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to initialize sweep. Please check your configuration."
    exit 1
fi

echo "Sweep initialized with ID: $SWEEP_ID"
echo "Starting $NUM_AGENTS agent(s)..."

# Start multiple agents if specified
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i..."
    CUDA_VISIBLE_DEVICES=$((i-1)) wandb agent "$PROJECT_NAME/$SWEEP_ID" &
done

echo "All agents started. Use 'jobs' to see running agents."
echo "To stop all agents, run: pkill -f 'wandb agent'"

# Wait for all background jobs
wait
