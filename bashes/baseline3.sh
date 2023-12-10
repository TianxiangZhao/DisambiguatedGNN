datasets=(computer)
methods=(DGI GRACE MVGRL SUP) #  GRACE MVGRL
losses=(jsd) # infonce
tasks=(pretrain joint) # pretrain
epoch=8000

for dataset in ${datasets[*]}
  do
    for method in ${methods[*]}
      do
        for loss in ${losses[*]}
          do
            for task in ${tasks[*]}
              do
                CUDA_VISIBLE_DEVICES=3 python BAL_main.py --dataset=$dataset --method=$method --loss=$loss --log --task=$task --epochs=$epoch
              done
          done
      done
  done