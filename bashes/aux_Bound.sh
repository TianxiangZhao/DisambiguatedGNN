datasets=(actor)
method=BOUND #  GRACE MVGRL
loss=jsd # infonce
task=joint
models=(gcn)

Boundrhos=(0.8)
Boundthreshs=(0.4 0.3 0.5 0.6 0.7 0.8)
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
                                #CUDA_VISIBLE_DEVICES=1 python deambiguous_main.py --dataset=$dataset  --model=$model --method=$method --loss=$loss --log --task=$task --epochs=$epoch --Boundrho=$Boundrho --Boundthresh=$Boundthresh --remote_K=$remote_K --aug_remote --Posthresh=$Posthresh --Negthresh=$Negthresh --seed=$seed --Boundweight=$weight
                                CUDA_VISIBLE_DEVICES=1 python deambiguous_main.py --dataset=$dataset  --model=$model --method=$method --loss=$loss --log --task=$task --epochs=$epoch --Boundrho=$Boundrho --Boundthresh=$Boundthresh --remote_K=$remote_K --Posthresh=$Posthresh --Negthresh=$Negthresh --seed=$seed --Boundweight=$weight
                              done
                          done
                      done
                  done
              done
          done
      done
  done