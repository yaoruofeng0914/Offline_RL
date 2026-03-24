## Implementation for Robust Decision Transformer

### Usage

1. **Generate the Downsampled Dataset:**
   ```
   cd utils
   python ratio_dataset.py --env_name walker2d-medium-replay-v2 --ratio 0.1
   ```

2. **Training:**

   **For Random State Corruption:**
   ```
   python algos/RDT.py \ 
   --seed 0 \
   --env walker2d-medium-replay-v2 \
   --corruption_mode random \
   --corruption_tag act \
   --save_model true \
   --eval_attack false
   ```

   **For Adversarial State Corruption:**
   ```
   python algos/RDT.py \ 
   --seed 0 \
   --env walker2d-medium-replay-v2 \
   --corruption_mode adversarial \
   --corruption_tag act \
   --save_model true \
   --eval_attack false
   ```

3. **Testing:**

   ```
   python algos/RDT.py \ 
   --env walker2d-medium-replay-v2 \
   --checkpoint_dir ./results/RDT/walker2d-medium-replay-v2/your_checkpoint_dir \
   --eval_only true \
   --eval_attack true
   ```

4. **Get the final testing results:**

   ```
   python process_results.py ./results/RDT
   ```
