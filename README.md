# e3sm_co2_diag
Helper code for evaluating CO2 in E3SM


## ⚙ Installation

###### 👥  Clone this repository to the location of your choice

(Most work will be done in the 'develop' branch, which we `checkout` before pulling the latest version)
```shell script
git clone https://github.com/dkauf42/e3sm_co2_diag.git ~/e3sm_co2_diag
cd ~/e3sm_co2_diag/
git checkout develop
git pull
```

###### 🌍  Create conda environment and install dependencies:
```shell script
conda create -n "myenv" python=3.8
conda config --add channels conda-forge
conda install --file requirements.txt
```

###### 💾  Install the package:
```shell script
pip install .
```

## Usage

Check out the demonstration notebooks in `notebooks/demo/` 
for how to run recipes for CMIP6 model output, NOAA Globalview+ Obspack, and E3SM model output.

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