@echo off
rem ------------------------------------------------------------------
rem set_path_vars, version 1 (for Windows)
rem
rem  This script sets the variables that GDESS accesses
rem+  when looking for data filepaths and where to save outputs.
rem
rem Note that the value of the GDESS_REPO environment variable must
rem+ be set prior to calling this script by running in the command 
rem+ line:
rem+ set GDESS_REPO=path_to_gdess
rem
rem Author: O'Meara, S.P.
rem Creation date: April 2022
rem ------------------------------------------------------------------
set GDESS_CMIP_DATA=${GDESS_REPO}/tests/test_data/cmip/
set GDESS_GLOBALVIEW_DATA=${GDESS_REPO}/tests/test_data/globalview/
set GDESS_SAVEPATH=${GDESS_REPO}/outputs/