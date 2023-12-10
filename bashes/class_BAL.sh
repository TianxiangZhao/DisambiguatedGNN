
datasets=(computer photo cora_full pubmed) #

methods=(DGI GRACE MVGRL SUP) #  GRACE MVGRL SUP
loss=jsd # infonce
tasks=(pretrain joint)

Sel_weights=(0.5)
Ent_weights=(0.5)
Class_weights=(2)
epoch=8000
seeds=(40 42 24)

for dataset in ${datasets[*]}
  do
    for method in ${methods[*]}
      do
        for seed in ${seeds[*]}
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

                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --do-map --BAL_weights ${weights[*]} --SUPBAL
                            CUDA_VISIBLE_DEVICES=1 python BAL_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --SUPBAL --BAL_weights ${weights[*]}  --debias --classwise --seed=$seed
                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --do-map --SUPBAL --BAL_weights ${weights[*]}
                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch  --SUPBAL --BAL_weights ${weights[*]}
                          done
                      done
                  done
              done
          done
      done
  done


methods=(DGI GRACE MVGRL) #  GRACE MVGRL
tasks=(pretrain joint) # pretrain
epoch=8000

for dataset in ${datasets[*]}
  do
    for method in ${methods[*]}
      do
        for seed in ${seeds[*]}
          do
            for task in ${tasks[*]}
              do
                CUDA_VISIBLE_DEVICES=1 python BAL_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --classwise --seed=$seed
              done
          done
      done
  done