
#!/bin/sh

current_time=$(date "+%Y.%m.%d-%H.%M")
for i in `seq 0 10`
do
cat <<EOS | qsub -
#!/bin/sh

#PBS -l nodes=1:ppn=1:gpu,walltime=1:00:00
#PBS -j oe
#PBS -o $PWD/$current_time$(printf noleaders%02d.txt $i)
export PATH=/usr/local/cuda/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cd $PWD
./runSwitcher $i
EOS
done
