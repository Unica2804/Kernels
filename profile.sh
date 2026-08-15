#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./profile.sh <executable_path>"
    exit 1
fi

EXE=$1
echo "-------------------------------------------------------"
echo "🚀 Profiling: $EXE"
echo "-------------------------------------------------------"

# Defining metrics for RTX 3050 (sm_86)
METRICS="gpu__time_duration.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

# Run profiling and capture the CSV
# We use --csv to get raw data
ncu --metrics $METRICS --csv "$EXE" > raw_profile.csv

# Helper function to extract the last value of a metric from the CSV
get_metric() {
    grep "$1" raw_profile.csv | tail -n 1 | awk -F',' '{print $NF}' | tr -d '"' | sed 's/ //g'
}

# Extract values
DURATION_NS=$(get_metric "gpu__time_duration.sum")
FADD=$(get_metric "fadd")
FMUL=$(get_metric "fmul")
FFMA=$(get_metric "ffma")
SM_UTIL=$(get_metric "sm__throughput")
MEM_UTIL=$(get_metric "gpu__compute_memory_throughput")

# Default to 0 if empty to prevent awk errors
DURATION_NS=${DURATION_NS:-0}
FADD=${FADD:-0}
FMUL=${FMUL:-0}
FFMA=${FFMA:-0}

# Calculate Stats
TOTAL_FLOPS=$(awk "BEGIN {print $FADD + $FMUL + (2 * $FFMA)}")
GFLOPS=$(awk "BEGIN {if($DURATION_NS > 0) print ($TOTAL_FLOPS / $DURATION_NS); else print 0}")
DURATION_MS=$(awk "BEGIN {print $DURATION_NS / 1000000}")

echo "📊 PERFORMANCE REPORT:"
echo "⏱️  Kernel Duration: ${DURATION_MS} ms"
echo "🧮 Total operations: ${TOTAL_FLOPS} FLOPs"
echo "⚡ Performance:     ${GFLOPS} GFLOPS"
echo "🧠 SM Throughput:   ${SM_UTIL}%"
echo "💾 Mem Throughput:  ${MEM_UTIL}%"
echo "-------------------------------------------------------"

rm raw_profile.csv