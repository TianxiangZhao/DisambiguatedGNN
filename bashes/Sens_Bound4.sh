datasets=(computer photo)
method=BOUND #  GRACE MVGRL
loss=jsd # infonce
task=joint
models=(gcn)

Boundrhos=(0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95)
Boundthreshs=(0.5)
remote_K=5
Posthreshs=(0.7)
Negthreshs=(0.4)
weights=(0.001)
epoch=8000
seeds=(4 43)

for dataset in ${datasets[*]}
  do
    for seed in ${seeds[*]}
      do
        for model in ${models[*]}
          do
            for Boundthresh in ${Boundthreshs[*]}
              do
                for Boundrho in ${Boundrhos[*]}
                  do
                    for Posthresh in ${Posthreshs[*]}
                      do
                        #CUDA_VISIBLE_DEVICES=0 python deambiguous_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --Boundrho=$Boundrho --Boundthresh=$Boundthresh
                          
                        for Negthresh in ${Negthreshs[*]}
                          do
                            for weight in ${weights[*]}
                              do
                                #CUDA_VISIBLE_DEVICES=2 python deambiguous_main.py --dataset=$dataset --method=$method --Boundweight=$weight --loss=$loss --log --task=$task --epochs=$epoch --Boundrho=$Boundrho --Boundthresh=$Boundthresh --remote_K=$remote_K --aug_remote --Posthresh=$Posthresh --Negthresh=$Negthresh --seed=$seed --model=$model
                                
                                CUDA_VISIBLE_DEVICES=3 python deambiguous_main.py --dataset=$dataset --method=$method --Boundweight=$weight --loss=$loss --log --task=$task --epochs=$epoch --Boundrho=$Boundrho --Boundthresh=$Boundthresh --remote_K=$remote_K --Posthresh=$Posthresh --Negthresh=$Negthresh --seed=$seed --model=$model
                              done
                          done
                      done
                  done
              done
          done
      done
  done