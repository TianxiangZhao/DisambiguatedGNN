datasets=(cora) #blogcatalog
methods=(SUP) #  GRACE MVGRL SUP DGI 
loss=jsd # infonce
tasks=(joint)
models=(gcn sage)

seeds=(4 43 24)
epoch=8000
weight=1


for dataset in ${datasets[*]}
  do
    for method in ${methods[*]}
      do
        for model in ${models[*]}
          do
            for task in ${tasks[*]}
              do
                for seed in ${seeds[*]}
                  do
                        CUDA_VISIBLE_DEVICES=0 python deambiguous_main.py --dataset=$dataset  --method=$method --Boundweight=$weight --loss=$loss --log --task=$task --epochs=$epoch --model=$model --seed=$seed
                          
                  done
              done
          done
      done
  done