datasets=(pubmed)
methods=(DGI GRACE MVGRL) #  GRACE MVGRL
losses=(jsd) # infonce
tasks=(pretrain joint)

Sel_weights=(0.5 1 2)
Ent_weights=(0 0.5 1)
Class_weights=(2)
epoch=8000

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

                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --do-map --BAL_weights ${weights[*]} --SUPBAL
                            CUDA_VISIBLE_DEVICES=3 python BAL_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --SUPBAL --BAL_weights ${weights[*]}  --debias
                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --do-map --SUPBAL --BAL_weights ${weights[*]}
                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch  --SUPBAL --BAL_weights ${weights[*]}
                          done
                      done
                  done
              done
          done
      done
  done