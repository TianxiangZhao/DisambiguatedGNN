# vanilla version, no reweight
datasets=(cora_full)
methods=(SUPBAL) #  GRACE MVGRL
losses=(jsd) # infonce
tasks=(joint)

Sel_weights=(0 0.5 1 2)
Ent_weights=(0 0.5 1 2)
Class_weights=(0 0.5 1 2)

epoch=4000



for dataset in ${datasets[*]}
  do
    for method in ${methods[*]}
      do
        for loss in ${losses[*]}
          do
            for task in ${tasks[*]}
              do
                for sel_weight in ${Sel_weights[*]}
                  do
                    for ent_weight in ${Ent_weights[*]}
                      do
                        for class_weight in ${Class_weights[*]}
                          do
                            weights=($sel_weight $ent_weight $class_weight)

                            #CUDA_VISIBLE_DEVICES=2 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --do-map --BAL_weights ${weights[*]}
                            CUDA_VISIBLE_DEVICES=2 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --BAL_weights ${weights[*]}
                            #CUDA_VISIBLE_DEVICES=2 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --do-map --BAL_weights ${weights[*]}
                            #CUDA_VISIBLE_DEVICES=2 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --BAL_weights ${weights[*]}
                          done
                      done
                  done
              done
          done
      done
  done