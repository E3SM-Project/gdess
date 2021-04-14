# e3sm_co2_diag
Generate diagnostics to help evaluate atmospheric CO<sub>2</sub> 
as simulated by the Energy Exascale Earth System Model (E3SM)

## Usage

Check out the demonstration notebooks in `notebooks/demo/` 
for how to run recipes for CMIP6 model output, NOAA Globalview+ Obspack, and E3SM model output.


## Installation

👥 Clone this repository to the location of your choice. 
*Note: most work will be done in the 'develop' branch, 
which we `checkout` before pulling the latest version.*
```shell script
git clone https://github.com/dkauf42/e3sm_co2_diag.git ~/e3sm_co2_diag
cd ~/e3sm_co2_diag/
git checkout develop
git pull
```

🌍 Create conda environment and install dependencies. 
*Note: Replace “myenv” with the preferred name of your environment, e.g. "e3sm_co2_diagnostics". 
From here on we’ll use “myenv” to refer to our environment.*

```shell script
conda create -n "myenv" python=3.8
conda activate "myenv"
conda config --add channels conda-forge
conda install --file requirements.txt
```

💾 Install the package:
```shell script
pip install .
```

## Updating

To use the latest version of this repository:
- Enter the `e3sm_co2_diag/` directory
- Activate your desired environment
- Run the commands:

   ```
   git pull
   git checkout main
   pip install . --upgrade
   ```

## Uninstalling

🚮 To remove this package from your environment:

```
pip uninstall co2_diag
```

## 📁 Project Structure

#### Components

<img src="./.images/structure_diagram_20210409.png" alt="components" width="607" height="384"/>

#### Directory Tree
```
e3sm_co2_diag
│
├── README.md                <- Top-level README for users/developers of this project
├── requirements.txt         <- Package dependencies
│
├── notebooks                <- Example jupyter notebooks to see diagnostic capabilities of co2_diag
│   └──demo/
│
├── co2_diag                 <- *Python package* for handling co2 diagnostics
│   │
│   ├── recipes              <- Generate repeatable diagnostics that span multiple data sources available as recipes 
│   │   ├── surface_trends.py
│   │   ├── utils.py
│   │   └── ...
│   │
│   ├── data_source          <- Modules to load, parse, and manipulate data from a particular source
│   │   ├── cmip/
│   │   ├── e3sm/
│   │   ├── obspack/
│   │   ├── datasetdict.py
│   │   ├── multiset.py
│   │   └── ...
│   │
│   ├── operations           <- Methods for manipulating datasets (e.g. spatially or temporally) 
│   │   ├── geographic/
│   │   ├── time/
│   │   ├── convert/
│   │   └── ...
│   │
│   ├── formatters           <- Manipulate formatting in desired ways
│   │   ├── nums.py
│   │   ├── strings.py
│   │   └── ...
│   │
│   ├── graphics             <- Make repeated graphic actions available 
│   │   ├── mapping.py
│   │   ├── utils.py
│   │   └── ...
│   │
│   └── config               <- Configuration options
│       └── log_config.json
│
│
├── MANIFEST.in
└── setup.py
```

## Credits

Major dependencies:

* [pandas](https://pandas.pydata.org/)
* [NumPy](https://www.numpy.org)
* [xarray](http://xarray.pydata.org/en/stable/)

Although not a dependency, ideas were also drawn from [xclim: Climate indices computations](https://github.com/Ouranosinc/xclim).

Funding Acknowledgment:

* Pacific Northwest National Laboratory

## Disclaimer

This is a work in progress.  Bugs are expected.