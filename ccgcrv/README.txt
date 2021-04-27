This directory contains the source code and executables for the ccgcrv program 
for applying curve fitting and filtering of time series data as explained at
https://www.esrl.noaa.gov/gmd/ccgg/mbl/crvfit/crvfit.html

There are two versions of the curve fitting code; C and python.  

C Version:
	The C code is in the file ccgcrv.tar.gz.  Untar this file then run 'make' to compile 
	the code and create the ccgcrv executable.

Python Version:
	The python code consists of 3 files:
	   ccgcrv.py - driver program
	   ccg_dates.py - module needed by ccgcrv.py
	   ccg_filter.py - module that does the curve fitting/filtering.

	Run ccgcrv.py with the --help option to see all available options.  The documentation
	of the ccg_filter.py module is in the pdf file ccgfilt.pdf

	The python code should run under either python 2.6+ or python 3.  It requires that
	the numpy, scipy and dateutil python packages are installed.

Graphical Program:
	Included in the zip file 'ccgvu.zip' are the files for running a graphical
	interface for ccgcrv call 'ccgvu', which allows you to select the files to process and 
	plots the resulting curves.  This progam uses the wxPython widget set, so this and
	all its dependencies will need to be installed.

Files:

	ccgcrv.exe - older Windows executable.  Not guaranteed to run.
	ccgcrv.py - Driver program for python version of ccgcrv
	ccgcrv.tar.gz - Tar file containing C code and makefile
	ccg_dates.py - module needed by ccgcrv.py
	ccgfilt.pdf - documentation of ccgfilt.py
	ccg_filter.py - module for performing curve fitting and filtering
	ccgvu.zip - zip file with code for a graphical user interface for
		using ccgcrv.  Requires wxPython. This uses the python module for
		curve fitting.  Unzip the file, then
		run ccgvu.py to start the program.  There is a
		test data file called 'mlotestdata.txt' that you can use to
		test the program.  Use the 'Import' menu function to read in
		this data file.

For any question on the code, contact kirk.w.thoning@noaa.gov.
