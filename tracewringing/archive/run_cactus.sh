#!/bin/bash

workload=cactusADM
window=10000
clusters=2
thres=100
gap=1
length=10
blocksize=3
fp=$1
theta=1

#==========================================

mkdir -p logs
name=$workload\_$blocksize\_$fp
log=$name.log

printf "Wringing $workload ...\n"
printf "window=$window, clusters=$clusters, thres=$thres, "
printf "gap=$gap, length=$length, blocksize=$blocksize\n"
printf "Log saved to logs/$log\n"

stdbuf -o0 python run_pipeline.py --path workload/$workload --name $name --window-size $window --n_clusters $clusters --iptype sqrt --threshold $thres --line_gap $gap --line_length $length --blocksize $blocksize --filter_percent $fp --theta_factor $theta > logs/$log

tar -czf $name.tar.gz labels/$name.labels hough_lines/$name.hls
comp_info="$(wc -c < $name.tar.gz)"
let "info = $comp_info * 8"
rm $name.tar.gz
printf "\nCompressed info (bits): $info\n" >> logs/$log

echo "Cache simulating..."
declare -a cache_size=("8192" "16384" "32768")
declare -a cache_assoc=("1" "4")
run=0
for size in "${cache_size[@]}"
do
    for assoc in "${cache_assoc[@]}"
    do
        sfile=$name\_$run.stats
        printf "Cache sim $run : cache size: $size assoc: $assoc\n" > logs/$sfile
        printf "\n======Original Trace======\n" >> logs/$sfile
        printf "python cachesim.py workload/$workload $size $assoc >> logs/$sfile\n"
        python cachesim.py workload/$workload $size $assoc >> logs/$sfile
        printf "\n======Proxy Trace======\n" >> logs/$sfile
        printf "python cachesim.py workload/proxy/$name $size $assoc >> logs/$sfile\n"
        python cachesim.py workload/proxy/$name $size $assoc >> logs/$sfile
        ((run++))
    done
done

printf "$workload done."
