# Update Instructions for Jupyter Notebook

Since Jupyter Notebook files (`.ipynb`) are often modified interactively during active training, the print statement formatting inside `PCB_Router_Training.ipynb` should be updated to align with the command-line training output (`train.py`).

Please update the print block in the **Training Loop** cell (usually Step 3) of your notebook to match the updated formatting.

## What to Change

Locate the print block at the bottom of the training loop inside `try...except KeyboardInterrupt` block:

### Old Print Statement
```python
        monitor.update(record)
        print(f"steps {steps_done:>8}  stage {stage}  ep_return {record['ep_return']:8.2f}  "
              f"completion {record['completion']:5.1%}  entropy {upd['entropy']:6.3f}  "
              f"pi {upd['pi_loss']:+.4f}  v {upd['v_loss']:8.3f}  clip {upd['clip_frac']:5.1%}  "
              f"drc {drc_total}  commit_rate {stats['commit_rate']:5.1%}  "
              f"{record['steps_per_sec']:6.0f} steps/s")
```

### New Print Statement
Replace it with the following:
```python
        monitor.update(record)
        total_nets = STAGES[min(stage, len(STAGES)-1)].n_nets
        mean_nets_routed = record['completion'] * total_nets
        print(f"steps {steps_done:>8}  stage {stage}  ep_return {record['ep_return']:8.2f}  "
              f"completion {record['completion']:5.1%} ({mean_nets_routed:.2f}/{total_nets} nets)  entropy {upd['entropy']:6.3f}  "
              f"pi {upd['pi_loss']:+.4f}  v {upd['v_loss']:8.3f}  clip {upd['clip_frac']:5.1%}  "
              f"drc {drc_total}  commit_rate {stats['commit_rate']:5.1%}  "
              f"{record['steps_per_sec']:6.0f} steps/s")
```

After updating, the console output will properly display the average number of successfully connected nets alongside the completion rate (e.g. `completion 100.0% (3.00/3 nets)`).
