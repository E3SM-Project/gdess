# e3sm_co2_diag
Helper code for evaluating CO<sub>2</sub> in E3SM

## Usage

Check out the demonstration notebooks in `notebooks/demo/` 
for how to run recipes for CMIP6 model output, NOAA Globalview+ Obspack, and E3SM model output.


## Installation

###### 👥  Clone this repository to the location of your choice

(Most work will be done in the 'develop' branch, which we `checkout` before pulling the latest version)
```shell script
git clone https://github.com/dkauf42/e3sm_co2_diag.git ~/e3sm_co2_diag
cd ~/e3sm_co2_diag/
git checkout develop
git pull
```

###### 🌍  Create conda environment and install dependencies:

💥 Important: Replace “myenv” with the preferred name of your environment, e.g. "e3sm_co2_diagnostics". 
From here on we’ll always use “myenv” to refer to our environment.

```shell script
conda create -n "myenv" python=3.8
conda activate "myenv"
conda config --add channels conda-forge
conda install --file requirements.txt
```

###### 💾  Install the package:
```shell script
pip install .
```

## Uninstalling

## Credits

Major dependencies:

* [pandas](https://pandas.pydata.org/)
* [NumPy](https://www.numpy.org)
* [xarray](http://xarray.pydata.org/en/stable/)

Funding Acknowledgment:

* Pacific Northwest National Laboratory

## Disclaimer

This is a work in progress.  Bugs are expected.