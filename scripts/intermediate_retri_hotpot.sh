#!/usr/bin/env bash

if [[ -z "$DIR_TMP" ]]; then    # If project root not defined.
    # get the directory of this file
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/.."; pwd)
fi

export DIR_TMP=$(cd "${DIR_TMP}"; pwd)
echo "The path of project root: ${DIR_TMP}"

# Download the data
# check if data exist.
if [[ ! -d ${DIR_TMP}/data ]]; then
    mkdir ${DIR_TMP}/data
fi

# download aux_file.
cd ${DIR_TMP}/data
if [[ ! -d vocab_cache ]]; then
    wget -O "p_hotpotqa.zip" "https://www.dropbox.com/s/sli7rv67kx4ud8w/p_hotpotqa.zip?dl=0"
    unzip "p_hotpotqa.zip" && rm "p_hotpotqa.zip" && rm "__MACOSX"
fi