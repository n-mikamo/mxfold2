#!/bin/sh
#$ -l gpu=1 -l h=hpc58 -l h_vmem=64G

# asdf
#if [ -f ${HOME}/.asdf/asdf.sh ]; then
#    . $HOME/.asdf/asdf.sh
#fi

# conda
#if [ -n "$ASDF_DIR" -a -f "$(asdf where python)/etc/profile.d/conda.sh" ]; then
#     . "$(asdf where python)/etc/profile.d/conda.sh"
#fi
. ${HOME}/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
conda activate mxfold2

#git show -s > version
#ln -sf ../../data .

common_opts="--gpu ${GPU} --epochs 100 --disable-progress-bar --seed 1234"
#common_opts="--gpu ${GPU} --epochs 50 --seed 1234"

for tr in TrainSetA; do
  mkdir -p $tr
  poetry run mxfold2 train ${common_opts} --log-dir $tr --save-config $tr.conf --param $tr.pth  \
	--model MixC --loss-func hinge_mix --dropout-rate 0.5 \
    --fc-dropout-rate 0.5 --l2-weight 0.01 --score-loss-weight 0.250 \
	--embed-size 64 \
    --num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3 \
	--num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3 \
    --num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3 \
    --num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3 \
	--num-lstm-layers 2 --num-lstm-units 32 --num-att 8 \
	--pair-join cat \
    --num-paired-filter 64 --paired-filter-size 5 --num-paired-filter 64 --paired-filter-size 3 \
	--num-paired-filter 64 --paired-filter-size 5 --num-paired-filter 64 --paired-filter-size 3 \
    --num-paired-filter 64 --paired-filter-size 5 --num-paired-filter 64 --paired-filter-size 3 \
    --num-paired-filter 64 --paired-filter-size 5 --num-paired-filter 64 --paired-filter-size 3 \
	data/$tr.lst

  for te in TrainSetA TestSetA TestSetB; do
    mkdir -p $tr-$te
    poetry run mxfold2 predict --gpu ${GPU} @$tr.conf --bpseq $tr-$te --result $tr-$te.res data/$te.lst
    echo
    echo $tr-$te:
    python ../avg-res.py $tr-$te.res
  done
done
