#!/bin/bash

for file in $1/*; do
    mkdir -p $1/logs
    mkdir -p $1/proxy
    name=$(basename $file)
    echo $name
    python run_pipeline.py --path $file --proxypath $1/proxy/ --name $name --window-size 100 --n_clusters 2 --collapse_factor 1 --iptype sqrt --threshold 10 --line_gap 10 --line_length 5 --blocksize 1 --height 2048 --theta_factor 1 --fast > $1/logs/$name
done
