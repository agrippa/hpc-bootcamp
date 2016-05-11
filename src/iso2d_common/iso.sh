#!/bin/bash

SCRIPT_DIR="$(dirname $0)"

NX=
NY=
INPUT=

OPTIND=1
while getopts "x:y:i:h" opt; do
    case "$opt" in
        x)  NX=$OPTARG
            ;;
        y)  NY=$OPTARG
            ;;
        i) INPUT=$OPTARG
            ;;
        h)
            echo 'usage: iso.sh -x xsize -y ysize -i input-file'
            exit 1
            ;;
    esac
done

if [[ -z "$NX" ]]; then
    echo 'x dimensionality must be passed using -x'
    exit 1
fi
if [[ -z "$NY" ]]; then
    echo 'y dimensionality must be passed using -y'
    exit 1
fi
if [[ -z "$INPUT" ]]; then
    echo 'input file must be passed using -i'
    exit 1
fi


gnuplot -e "INPUT='$INPUT'" -e "NX=$NX" -e "NY=$NY" $SCRIPT_DIR/iso.gnu
