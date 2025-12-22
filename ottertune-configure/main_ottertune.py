#!/usr/bin/env python
import os
import sys
import numpy as np
import json
import time
import subprocess as sp
import pty
import random

# Add paths to sys.path
OTTERTUNE_SERVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ottertune/server'))
AUTO_CONFIGURE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../auto-configure'))
VDTUNER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../auto-configure/vdtuner'))

if OTTERTUNE_SERVER_PATH not in sys.path:
    sys.path.append(OTTERTUNE_SERVER_PATH)
if AUTO_CONFIGURE_PATH not in sys.path:
    sys.path.append(AUTO_CONFIGURE_PATH)
if VDTUNER_PATH not in sys.path:
    sys.path.append(VDTUNER_PATH)

from analysis.gp_tf import GPRGD
from configure import filter_index_rule, configure_index, filter_system_rule, configure_system
from utils import KnobStand, LHS_sample

# Configuration
KNOB_PATH = os.path.join(AUTO_CONFIGURE_PATH, 'whole_param.json')
RUN_ENGINE_PATH = '/home/dyx/VDTuner/vector-db-benchmark-master/run_engine_test.sh'
LOG_FILE = 'ottertune.log'

def run_benchmark(config, knob_names, knob_stand):
    """
    Apply configuration and run benchmark.
    Returns: (rps, precision, p95_time)
    """
    # Separate index and system parameters
    index_keys = ['index_type', 'nlist', 'nprobe', 'm', 'nbits', 'M', 'efConstruction', 'ef', 'reorder_k']
    
    # Scale back knobs
    real_conf = {}
    for i, val in enumerate(config):
        name = knob_names[i]
        _, real_val = knob_stand.scale_back(name, val)
        real_conf[name] = real_val

    index_conf = {k: real_conf[k] for k in index_keys if k in real_conf}
    system_conf = {k: real_conf[k] for k in real_conf if k not in index_keys}

    print(f"[OtterTune] Applying config: {index_conf} {system_conf}")
    configure_index(*filter_index_rule(index_conf))
    configure_system(filter_system_rule(system_conf))

    # Run the benchmark
    run_cmd = f'timeout 900 {RUN_ENGINE_PATH} "" "" glove-100-angular'
    print(f"[OtterTune] Running benchmark...")
    
    try:
        master, slave = pty.openpty()
        process = sp.Popen(run_cmd, shell=True, stdout=slave, stderr=slave, close_fds=True)
        os.close(slave) 

        stdout_lines = []
        while True:
            try:
                output = os.read(master, 1024)
                if not output: break
            except OSError: break
            text = output.decode('utf-8', errors='replace')
            print(text, end='', flush=True)
            stdout_lines.append(text)
        
        process.wait()
        stdout_str = "".join(stdout_lines)
        
        # Parse metrics
        result_lines = stdout_str.split()
        metrics = []
        for val in reversed(result_lines):
            try:
                metrics.append(float(val.strip(',')))
                if len(metrics) == 3: break
            except ValueError: continue
        
        if len(metrics) == 3:
            # metrics: p95_time, rps, mean_precisions
            return metrics[1], metrics[2], metrics[0]
        else:
            print("[OtterTune] Failed to parse metrics from output.")
            return None
    except Exception as e:
        print(f"[OtterTune] Error running benchmark: {e}")
        return None

def main():
    print("--- OtterTune VDTuner Integration ---")
    
    # 1. Initialize KnobStand and get knob names
    knob_stand = KnobStand(KNOB_PATH)
    knob_names = list(knob_stand.knobs_detail.keys())
    num_knobs = len(knob_names)
    
    # 2. Initial sampling (LHS)
    NUM_INITIAL_SAMPLES = 5
    print(f"Generating {NUM_INITIAL_SAMPLES} initial samples using LHS...")
    X_train = LHS_sample(num_knobs, NUM_INITIAL_SAMPLES, seed=int(time.time()))
    y_train = []
    
    valid_X = []
    valid_y = []

    for i in range(NUM_INITIAL_SAMPLES):
        print(f"\n--- Initial Sample {i+1}/{NUM_INITIAL_SAMPLES} ---")
        res = run_benchmark(X_train[i], knob_names, knob_stand)
        if res:
            rps, precision, p95 = res
            # Score: we want to maximize RPS, so we minimize -RPS.
            # However, GPRGD works best with y in [0, 1].
            # We also penalize low precision.
            score = -rps if precision >= 0.9 else 0.0 # Placeholder, will scale later
            valid_X.append(X_train[i])
            valid_y.append(score)
            print(f"Result: RPS={rps}, Precision={precision}, Score={score}")
        else:
            print("Sample failed, skipping.")

    if not valid_X:
        print("No valid initial samples. Exiting.")
        return

    # Convert to numpy arrays
    X_train = np.array(valid_X, dtype=np.float32)
    y_train = np.array(valid_y, dtype=np.float32).reshape(-1, 1)

    # 3. Iterative Tuning
    MAX_ITER = 20
    X_min_bound = np.zeros(num_knobs, dtype=np.float32)
    X_max_bound = np.ones(num_knobs, dtype=np.float32)

    for iter_idx in range(MAX_ITER):
        print(f"\n\n--- OtterTune Iteration {iter_idx+1}/{MAX_ITER} ---")
        
        # Scale y to [0, 1] for GPRGD
        y_min = np.min(y_train)
        y_max = np.max(y_train)
        if y_max == y_min:
            y_scaled = np.zeros_like(y_train)
        else:
            y_scaled = (y_train - y_min) / (y_max - y_min)

        # Initialize GPRGD model
        # Parameters inspired by simple_ottertune.py
        model = GPRGD(length_scale=1.0, magnitude=1.0, ridge=0.1, max_iter=50, learning_rate=0.001)
        
        # Fit model
        model.fit(X_train, y_scaled, X_min=X_min_bound, X_max=X_max_bound)
        
        # Generate test points for starting gradient descent
        # We'll use some random points and the current best
        num_test_points = 10
        X_test = np.random.uniform(0, 1, (num_test_points, num_knobs)).astype(np.float32)
        best_idx = np.argmin(y_train)
        X_test = np.vstack([X_test, X_train[best_idx]])
        
        # Add some jitter to avoid starting exactly at a training point (avoid sqrt(0) in gradients)
        X_test += np.random.normal(0, 0.01, X_test.shape).astype(np.float32)
        X_test = np.clip(X_test, 0, 1)

        # Recommend new configuration
        print("Searching for next best configuration...")
        results = model.predict(X_test)
        new_config = results.minl_conf[0] # Pick the best one from GD results

        # Run benchmark with new config
        res = run_benchmark(new_config, knob_names, knob_stand)
        if res:
            rps, precision, p95 = res
            score = -rps if precision >= 0.9 else 0.0
            
            # Update training data
            X_train = np.vstack([X_train, new_config])
            y_train = np.vstack([y_train, [score]])
            
            print(f"Iteration {iter_idx+1} Result: RPS={rps}, Precision={precision}, Score={score}")
            
            with open(LOG_FILE, 'a') as f:
                f.write(f"Iter {iter_idx+1}: RPS={rps}, Prec={precision}, Score={score}\n")
        else:
            print("Iteration failed.")

    print("\n--- Tuning Finished ---")
    best_idx = np.argmin(y_train)
    print(f"Best RPS found: {-y_train[best_idx][0]}")
    # ... could print best config here ...

if __name__ == '__main__':
    main()
