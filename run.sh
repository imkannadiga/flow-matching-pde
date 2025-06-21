#!/bin/bash

# Exit on error if a command fails
set -e

# Check if at least one task is passed
if [ "$#" -eq 0 ]; then
    echo "Usage: ./run.sh task1 task2 ..."
    exit 1
fi

# Loop over each provided task name
for task_name in "$@"; do
    task_file="tasks/${task_name}/task.py"

    if [[ -f "$task_file" ]]; then
        echo "Starting $task_file in background..."
        python3 "$task_file" &
    else
        echo "Task $task_name not found or missing task.py"
    fi
done

echo "All specified tasks have been started in the background."
