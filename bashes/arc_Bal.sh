datasets=(photo computer cora_full reddit pubmed)
methods=(DGI) #  GRACE MVGRL
loss=jsd # infonce
models=(sage gin)
tasks=(joint)

Sel_weights=(0.5 1 2)
Ent_weights=(0 0.5 1)
Class_weights=(2) 

epoch=8000



for dataset in ${datasets[*]}
  do
    for method in ${methods[*]}
      do
        for model in ${models[*]}
          do
            for task in ${tasks[*]}
              do
                CUDA_VISIBLE_DEVICES=2 python BAL_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --model=$model

                for sel_weight in ${Sel_weights[*]}
                  do
                    for ent_weight in ${Ent_weights[*]}
                      do
                        for class_weight in ${Class_weights[*]}
                          do
                            weights=($sel_weight $ent_weight $class_weight)

                            #CUDA_VISIBLE_DEVICES=2 python BAL_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --SUPBAL --BAL_weights ${weights[*]} --debias --model=$model
                          done
                      done
                  done
              done
          done
      done
  done