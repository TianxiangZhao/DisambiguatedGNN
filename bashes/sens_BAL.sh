datasets=(cora_full)
methods=(DGI) #  GRACE MVGRL
loss=jsd # infonce
tasks=(joint)

Sel_weights=(2)
Ent_weights=(0.5)
Class_weights=(2)
groupnumbers=(1 1.5 2.5 3)
epoch=8000



for dataset in ${datasets[*]}
  do
    for method in ${methods[*]}
      do
        for groupnumber in ${groupnumbers[*]}
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
                            CUDA_VISIBLE_DEVICES=1 python BAL_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --normalize --SUPBAL --BAL_weights ${weights[*]} --debias --groupnumber=$groupnumber
                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch --do-map --SUPBAL --BAL_weights ${weights[*]}
                            #CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch  --SUPBAL --BAL_weights ${weights[*]}
                          done
                      done
                  done
              done
          done
      done
  done