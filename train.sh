#!/bin/bash

algos=("RDT")
envs=("door-expert-v0" "halfcheetah-medium-replay-v2" "hammer-expert-v0" "hopper-medium-replay-v2" "kitchen-complete-v0" "kitchen-mixed-v0" "kitchen-partial-v0" "relocate-expert-v0" "walker2d-medium-replay-v2")
corruption_modes=("adversarial" "random")
corruption_tags=("act" "obs" "rew")
seed=0

for algo in "${algos[@]}"; do
	for env in "${envs[@]}"; do
		for corruption_mode in "${corruption_modes[@]}"; do
			for corruption_tag in "${corruption_tags[@]}"; do
		
	    			echo "--------------------------------------------------"
	    			echo "algos=$algo, Env=$env, corruption_mode=$corruption_mode, Corruption_tag=$corruption_tag"
	    			echo "--------------------------------------------------"
	    		
	    			python -m algos."$algo" \
					--env "$env" \
					--corruption_mode "$corruption_mode" \
					--corruption_tag "$corruption_tag" \
					--seed "$seed" \
					--save_model true 
				
		
			done
		done
	done
done
