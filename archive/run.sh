#!/bin/bash

workload=gzip2.trace
window=1000
clusters=5
thres=50
gap=1
length=1
blocksize=3
percent=0
theta=1

#==========================================

mkdir -p logs
name=$workload\_$blocksize\_$percent
log=$name.log

printf "Wringing $workload ...\n"
printf "window=$window, clusters=$clusters, thres=$thres, "
printf "gap=$gap, length=$length, blocksize=$blocksize\n"
printf "Log saved to logs/$log\n"

gstdbuf -o0 python3 run_pipeline.py --path workload/$workload --name $name --window-size $window --n_clusters $clusters --iptype sqrt --threshold $thres --line_gap $gap --line_length $length --blocksize $blocksize --filter_percent $percent --theta_factor $theta > logs/$log

tar -czf $name.tar.gz labels/$name.labels hough_lines/$name.hls
comp_info="$(wc -c < $name.tar.gz)"
let "info = $comp_info * 8"
rm $name.tar.gz
printf "\nCompressed info (bits): $info\n" >> logs/$log

printf "Cache simulating...\n"
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
        python3 cachesim.py workload/$workload $size $assoc >> logs/$sfile
        printf "\n======Proxy Trace======\n" >> logs/$sfile
        printf "python cachesim.py workload/proxy/$name $size $assoc >> logs/$sfile\n"
        python3 cachesim.py workload/proxy/$name $size $assoc >> logs/$sfile
        ((run++))
    done
done

printf "$workload done.\n"
