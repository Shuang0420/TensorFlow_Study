#!/bin/bash


# input: query \t FAQ
# output: segmented query \t FAQid

#file name
if [ x$1 != x ] && [ x$2 != x ]
then
    input=$1
    output=$2
else
    echo "please run the following command ./segClass.sh input output"
    exit 12
fi

# UTF-8 --> GBK
enca -L zh_CN -x GBK $input

# split words use words-segmentation software which defines GBK-format input and output
cd qqseg_new
./SegTester --input_file=../$input --output_file=../out.seg
cd ..

# GBK --> UTF-8
enca -L zh_CN -x UTF-8 out.seg
enca -L zh_CN -x UTF-8 $input

# get output
python GetCheckFile.py $input o
awk -F'##' '{if ($2!="")print $0}' o > $output

# delete intermediate files
rm -rf $input
rm -rf out.seg
rm -rf o
