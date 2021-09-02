#!/bin/bash
# ------------------------------------------------------------------
# set_path_vars, version 1
#
#  This script sets the variables that GDESS accesses
#+  when looking for data filepaths and where to save outputs.
#
# Author: Kaufman, D.E.
# Creation date: August 2021
# ------------------------------------------------------------------
path_to_repo=${GDESS_REPO}  # For example: ${HOME}/gdess
echo "Path to repo has been set to <${path_to_repo}>"

export GDESS_CMIP_DATA=${path_to_repo}/tests/test_data/cmip/
export GDESS_GLOBALVIEW_DATA=${path_to_repo}/tests/test_data/globalview/

export GDESS_SAVEPATH=${path_to_repo}/outputs/
