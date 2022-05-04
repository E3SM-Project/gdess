#!/bin/bash

source ${path_to_env_script}/load_latest_e3sm_unified.sh

map_dir=${map_dir}
dir_in=${HOME}/E3SM_simulations/${run_name}/run
dir_out=${HOME}/e3sm_scratch/${run_name}/ppdata

map=${map_name}

# Output directory is created if it doesn't yet exist.
[ -d ${dir_out} ] || mkdir ${dir_out}

# 
vars_to_include=""
vars_to_include+="lat,lon,area,"  # geometry
vars_to_include+="CO2"  # CO2

file_out="output_file"

#
ncrcat -h -v ${vars_to_include} ${dir_in}/${run_name}.cam.h0.????-??.nc ${dir_out}/${file_out}.nc

#
ncremap -m ${map_dir}/${map} -i ${dir_out}/${file_out}.nc -o ${dir_out}/${file_out}_rgd.nc
