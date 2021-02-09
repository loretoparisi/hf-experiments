#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

infile=$1
dration=$2
infile_name="${infile%%.*}"
infile_ext="${infile#*.}"

[ -z "$1" ] && { echo "Usage: $0 input_file_path (mp3)"; exit 1; }


if [ -z "$1" ]; then
    echo "Please specify input audio file (mp3)"
    exit
fi
if [ -z "$2" ]; then
    duration=30
fi

echo "cutting ${infile_name} . ${infile_ext} to ${duration} seconds"
ffmpeg -hide_banner -loglevel error -y -i ${infile} -ss 00:00:00 -t $duration -async 1 ${infile_name}_tmp.mp3
echo "converting to wav"
ffmpeg -hide_banner -loglevel error -y -i ${infile_name}_tmp.mp3 -acodec pcm_s16le -ac 1 -ar 16000 ${infile_name}.wav
rm ${infile_name}_tmp.mp3
echo "wav file: ${infile_name}.wav"