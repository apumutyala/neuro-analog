import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run full architecture sweep across devices")
    parser.add_argument('--task', type=str, choices=['cifar10', 'wikitext2', 'both'], default='both')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='Comma-separated list of GPU IDs to use')
    args = parser.parse_args()

    architectures = ['transformer', 'neural_ode', 's4d', 'deq', 'flow', 'ebm', 'diffusion']
    
    tasks_to_run = []
    if args.task in ['cifar10', 'both']:
        for arch in architectures:
            tasks_to_run.append(('train_cifar10.py', arch, 'cifar10'))
            
    if args.task in ['wikitext2', 'both']:
        for arch in architectures:
            tasks_to_run.append(('train_wikitext2.py', arch, 'wikitext2'))

    print(f"Total jobs to run: {len(tasks_to_run)}")
    
    gpus = [int(g.strip()) for g in args.gpus.split(',') if g.strip()]
    if not gpus:
        print("No GPUs specified. Using CPU.")
        gpus = [-1]
        
    print(f"Available GPUs: {gpus}")

    active_processes = []  # List of (process, gpu_id, task_description)
    gpu_pool = gpus.copy()
    
    # Create logs directory
    Path("run_logs").mkdir(exist_ok=True)

    task_idx = 0
    while task_idx < len(tasks_to_run) or active_processes:
        # Check active processes
        for p, gpu, desc in active_processes[:]:
            if p.poll() is not None:
                # Process finished
                code = p.returncode
                status = "SUCCESS" if code == 0 else f"FAILED (Code: {code})"
                print(f"[{status}] Finished {desc} on GPU {gpu}")
                gpu_pool.append(gpu)
                active_processes.remove((p, gpu, desc))

        # Launch new processes if we have available GPUs and pending tasks
        while gpu_pool and task_idx < len(tasks_to_run):
            gpu = gpu_pool.pop(0)
            script, arch, exp_type = tasks_to_run[task_idx]
            task_idx += 1

            desc = f"{exp_type}/{arch}"

            # Skip jobs whose summary file already exists (completed in a prior run)
            if exp_type == 'cifar10':
                summary_file = Path(f'checkpoints/cifar10/{arch}_cifar10_summary.json')
            else:
                summary_file = Path(f'checkpoints/wikitext2/{arch}_lm_summary.json')
            if summary_file.exists():
                print(f"[SKIP] Already completed: {desc}")
                gpu_pool.append(gpu)
                continue

            print(f"[START] Launching {desc} on GPU {gpu}")
            
            env = os.environ.copy()
            if gpu != -1:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                
            log_file = open(f"run_logs/{exp_type}_{arch}_runner.log", "w")
            
            cmd = [sys.executable, script, '--arch', arch, '--device', 'cuda' if gpu != -1 else 'cpu']
            
            p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
            active_processes.append((p, gpu, desc))
            
        time.sleep(5)

    print("All tasks completed.")

if __name__ == '__main__':
    main()
