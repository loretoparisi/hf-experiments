#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

if [ -z "$cache_dir" ]; then
    cache_dir="../../models/"
fi

if [ ! -d $cache_dir ] 
then
    mkdir $cache_dir
fi

DATASET=$1

echo "Downlading $DATASET dataset to $cache_dir..."


# https://stackoverflow.com/a/67611180/758836
function curl2file() {
    url=$1
    url=$(curl -L --head -w '%{url_effective}' $url 2>/dev/null | tail -n1) ; curl -O $url
}

data_dir=$cache_dir
if [ "${DATASET}" == "supervised" ]; then
    rm -rf $data_dir/supervised
    mkdir $data_dir/supervised
    cd $data_dir/supervised
    curl2file https://cloud.tsinghua.edu.cn/f/4666d28af98a4e63afb5/?dl=1
    curl2file https://cloud.tsinghua.edu.cn/f/6293b3d54f954ef8a0b1/?dl=1
    curl2file https://cloud.tsinghua.edu.cn/f/ae245e131e5a44609617/?dl=1
    cd -
elif [ "${DATASET}" == 'inter' ]; then
    rm -rf $data_dir/inter
    mkdir $data_dir/inter
    cd $data_dir/inter
    curl2file https://cloud.tsinghua.edu.cn/f/eeec65751e3148af945e/?dl=1
    curl2file https://cloud.tsinghua.edu.cn/f/45d55face2a14c098a13/?dl=1
    curl2file https://cloud.tsinghua.edu.cn/f/9b529ee30f4544299bc2/?dl=1
    cd -
elif [ "${DATASET}" == 'intra' ]; then
    rm -rf $data_dir/intra
    mkdir $data_dir/intra
    cd $data_dir/intra
    curl2file https://cloud.tsinghua.edu.cn/f/9a1dc235abc746a6b444/?dl=1
    curl2file https://cloud.tsinghua.edu.cn/f/b169cfbeb90a48c1bf23/?dl=1
    curl2file https://cloud.tsinghua.edu.cn/f/997dc82d29064e5ca8de/?dl=1
    cd -
else
    echo "Usage: $0 supervised|inter|intra"; exit 1;
fi