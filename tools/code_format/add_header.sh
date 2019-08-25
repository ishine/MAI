#!/bin/bash -
#===============================================================================
#
#          FILE: add_header.sh
# 
#         USAGE: ./add_header.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 2019年08月24日 22:29
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

path=$1
headerPath=$2

fileSuffixes=(.java .h .cpp)

headerFirstLine=`head -n +1 ${headerPath}`

checkAndAddHeader() {
    file=$1
    needAddHeader=false
    while read line
    do
        if [ "$line" != "$headerFirstLine" ];then
            needAddHeader=true
            echo "Need add header for $file"
        fi
        break
    done < $file

    if [ "$needAddHeader" == "true" ];then
        newFile=${file}__.back
        cat ${headerPath} > ${newFile}
        cat ${file} >> ${newFile}
        mv ${newFile} ${file}
    fi
}

for fileSuffix in ${fileSuffixes[*]};do
    files=`find $path -name *$fileSuffix`
    for file in ${files[*]};do
        checkAndAddHeader $file
    done
done
