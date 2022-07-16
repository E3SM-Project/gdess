---
title: 'gdess: A framework for evaluating simulated atmospheric CO~2~ in Earth System Models'
tags:
  - Python
  - Earth system model
  - atmospheric carbon dioxide
  - model evaluation
  - diagnostics
authors:
  - name: Daniel E. Kaufman
    orcid: 0000-0002-1487-7298
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Sha Feng
    orcid: 0000-0002-2376-0868
    affiliation: 2
  - name: Katherine V. Calvin
    orcid: 0000-0003-2191-4189
    affiliation: 1
  - name: Bryce E. Harrop
    orcid: 0000-0003-3952-4525
    affiliation: 2
  - name: Susannah M. Burrows
    orcid: 0000-0002-0745-7252
    affiliation: 2
affiliations:
 - name: Joint Global Change Research Institute, Pacific Northwest National Laboratory, College Park, MD, USA
   index: 1
 - name: Atmospheric Sciences and Global Change Division, Pacific Northwest National Laboratory, Richland, WA, USA
   index: 2
date: 23 August 2021
bibliography: paper.bib
---

# Summary

Atmospheric carbon dioxide (CO~2~) plays a key role in the global carbon cycle and global warming. Climate-carbon feedbacks are often studied and estimated using Earth System Models (ESMs), which couple together multiple model components—including the atmosphere, ocean, terrestrial biosphere, and cryosphere—to jointly simulate mass and energy exchanges within and between these components.  Despite tremendous advances, model intercomparisons and benchmarking are aspects of ESMs that warrant further improvement [@Smith2014ams; @Fer2021gcb]. Such benchmarking is critical because comparing the value of state variables in these simulations against observed values provides evidence for appropriately refining model components; moreover, researchers can learn much about Earth system dynamics in the process [@Randall2019c].

We introduce `gdess` (a.k.a., Greenhouse gas Diagnostics for Earth System Simulations), which parses observational datasets and ESM simulation output, combines them to be in a consistent structure, computes statistical metrics, and generates diagnostic visualizations. In its current incarnation, `gdess` facilitates evaluating a model's ability to reproduce observed temporal and spatial variations of atmospheric CO~2~. The diagnostics implemented modularly in `gdess` support more rapid assessment and improvement of model-simulated global CO~2~ sources and sinks associated with land and ocean ecosystem processes. We intend for this set of automated diagnostics to form an extensible, open source framework for future comparisons of simulated and observed concentrations of various greenhouse gases across Earth system models.

# Statement of need

Thorough evaluation of simulated atmospheric CO~2~ concentrations—by comparing against observations—requires multiple diagnostics, metrics, and visualizations. During the past decade, such evaluations have utilized certain common methods, such as aggregating in situ measurements into latitude bands and detrending of multidecadal time series to investigate seasonal cycles [@KeppelAleks2013b; @Ott2015b; @Liptak2017b; @Jing2018b; @Chevallier2019e; @WeirEtAl2021acp]. However, the construction of diagnostics used in these evaluations has not been automated in an open-source tool available to the broader atmospheric modeling community. Thus, each modeling or analysis team has had to decide on and code their own preferred set of diagnostics, resulting in redundancies and potential inconsistencies among efforts.

Several software packages have been developed to streamline the application of diagnostics for ESM benchmarking. These tools share related functionality with `gdess`, and some have directly inspired the `gdess` design and our development approach. For example, the ESM Evaluation Tool (ESMValTool; @Eyring2016c; @Eyring2020b) has been used to generate specific figures from the literature, and we adopted the term *recipe* from its use by ESMValTool. Although ESMValTool includes a comparison of column-averaged CO~2~ values as performed by Gier et al. (2020), `gdess` was created to provide specific CO~2~ diagnostic methods and graphs that are not already provided as recipes in ESMValTool. `gdess` uses Observation Package (Obspack; @globalview20200911; @Masarie2014b) data, which include atmospheric greenhouse gas observations from a variety of sampling platforms and data providers following the World Data Centre for Greenhouse Gases (WDCGG) protocol, so are widely used for stimulating and supporting carbon cycle modeling studies.  These data have not been set up for use within ESMValTool, and as such would require additional development/configuration to work with ESMValTool. The International Land Model Benchmarking (ILAMB) System [@Collier2018a] excels at intercomparisons between multiple land models and has been used to benchmark inferred CO~2~ concentrations against surface station measurements [@WuEtAl2020jhydro]. In contrast to `gdess`, ILAMB provides the means to evaluate emulated results but not prognostic simulations for CO~2~ [@KeppelAleks2021PersComm].

# Design and data sources

`gdess` is written in Python ["version 3"; @python; @VanRossumAndDrake2009]. A comprehensive readme file and docstrings throughout the open source codebase (https://github.com/E3SM-Project/gdess) provide documentation and guidance, and Continuous Integration tests facilitate further code development and maintenance. Data variables are represented and handled in memory using xarray, an open-source Python package for working with labeled multi-dimensional arrays [@Hoyer2017xarray]. 

As shown in \autoref{fig:code_schematic}, `gdess` is organized into modular components. A *Collection* class encapsulates source-specific attributes and methods for each data source (described below) and each Collection inherits common attributes from a parent *Multiset* class. Each diagnostic recipe, defined in a separate module file (e.g., ``surface_trends.py``), instantiates and uses Collection objects to handle the loading and pre-processing of data. Additionally, visualization functions (e.g., time-series, annual cycles) are accessible from any instance of a Collection or Multiset so that data sources can be inspected individually—i.e., without the need to run one of the comparative diagnostic recipes.

![Schematic of the `gdess` code structure.\label{fig:code_schematic}](graphics/gdess_structure_schematic_20210816.png){ width=70% }

`gdess` can process data from three sources: Globalview+, CMIP, and E3SM. Data from surface observing stations must be retrieved from the NOAA Global Monitoring Laboratory (GML) Globalview+ version $6.0$ Observation package (Obspack; @globalview20200911; @Masarie2014b).  In situ and flask measurements can be used from approximately 200 stations whose data in Obspack spans at least a 12 month period (\autoref{fig:surface_station_map}).

![Global map showing surface observing station locations (red circles) and their three-letter site codes, as recorded in Obspack and used in `gdess`.\label{fig:surface_station_map}](graphics/surf_station_map.png){ width=65% }

We distinguish between the model results from two different sources: (i) simulations by the Energy Exascale Earth System Model (E3SM), and (ii) other Earth system models participating in the latest, Version 6, Coupled Model Intercomparison Project (CMIP6). E3SM is a global modeling system composed of multiple coupled subcomponent models: atmosphere, ocean, land, ice [@GolazEtAl2019james; @Burrows2020b]. In this study, our focus is on evaluating CO~2~ mole fractions in the atmospheric component, which is called the E3SM atmosphere model (EAM) and which has been described in detail by @Rasch2019d. 

CMIP6 organizes the setup, experimental design, and intercomparisons of simulations performed using numerous global climate models. Data from CMIP6 are accessed either via locally stored files—downloaded directly from Earth System Grid Federation (ESGF) data nodes—or programmatically via the *intake-esm* package, which is a `gdess` dependency maintained as part of the *Pangeo* project. By default, comparisons in `gdess` use data from the 'esm-hist' experiment, which contains CO~2~ emission-driven simulations that span the period of 1850 to 2014—i.e., an "all-forcing simulation of the recent past with atmospheric CO~2~ concentration calculated" [@Eyring2016d]. We expect model output from any CMIP6 experiment could be used by specifying the appropriate data identifier or file location, although additional testing would be needed to confirm expected behavior.

# Functionality

This section describes and provides example output from the three diagnostic recipes implemented in `gdess`. These recipes can be initiated either from a terminal or from within a running Python kernel. The command-line interface consists of the ``gdess`` command, followed by the type of recipe, and then options for each recipe—e.g., which observing station(s) to use for comparison. Within a Python kernel, options are specified via a dictionary object. 

##### Multidecadal trend

Skillful simulation of the historical multidecadal trend in atmospheric CO~2~ is a necessary condition for an ESM to be an effective tool for conducting climatological projections and analyses. The research questions one might address with this diagnostic recipe (see example output in \autoref{fig:surface_trend}) include: What are the long-term biases in the model simulation?  How does the simulated increase in CO~2~ mixing ratios compare to surface measurements?

![Example output of the ``surface_trends`` recipe, showing (a) individual time series and (b) differences between simulated and observed concentrations of surface-level atmospheric CO~2~ at the Mauna Loa Observatory, Hawaii (MLO).\label{fig:surface_trend}](graphics/trends/combined_20220715.svg){ width=90% }

##### Seasonal cycle

Because of the substantial impact primary production and respiration have on CO~2~ concentrations, evaluating the seasonal cycle at a given location can help disentangle the effects of biological from physical processes. The seasonal cycle can be quantified by "the projection of an atmospheric time series onto a suitably defined subset of orthogonal basis functions, the choice of which depends on the length of the series involved" [@Straus1983ams].  For computing the seasonal cycle, we detrend the time series by fitting a function composed of both polynomial and harmonic terms, following the procedure of @Sweeney2015b and originally proposed by @Thoning1989jgr. Example output of the ``seasonal cycle`` recipe is shown in \autoref{fig:seasonal_cycle}.

![Example output of the ``seasonal_cycle`` recipe, comparing annual climatologies of surface atmospheric CO~2~ concentrations at the American Samoa Observatory, Tutuila Island (SMO).\label{fig:seasonal_cycle}](graphics/seasonal/overlapped_2022-07-15.svg){ width=75% }

##### Meridional gradient

By comparing CO~2~ concentrations across observing sites distributed globally, we can assess whether simulated transport and mixing is skillfully reproducing spatial gradients. For instance, the surface CO~2~ flux signals at lower latitudes (30-45N) are moved to northern boreal latitudes and also to the south by large scale circulation. Spatial analysis can reveal evidence of southward movement toward (sub)tropical convection that becomes mixed with Hadley circulation or northward movement toward midlatitude synoptic weather patterns and the Ferrell circulation [@DenningEtAl1999tellusb; @StephensEtAl2007science; @SchuhEtAl2019gbc]. \autoref{fig:meridional} shows example output of the ``meridional`` recipe.

![Example output of the ``meridional`` recipe, comparing the seasonal cycle across latitudes, at locations of user-specified surface stations.\label{fig:meridional}](graphics/meridional/meridional_heatmap_2022-07-15.svg){ width=75% }

# Outlook

Currently, `gdess` is helping to assess simulations using the biogeochemistry configuration of E3SM, with the aim of exploring carbon-climate interactions. In addition to the three implemented recipes (multidecadal trends, seasonal cycles, and meridional gradients), current development includes two other methods—by which CO~2~ was also evaluated by @KeppelAleks2013b—vertical gradients and interannual variability. Future releases may evaluate vertical gradients using aircraft data from Globalview$+$ Obspack, include satellite data, and extend to data for other greenhouse gases, such as methane. 

# Acknowledgements

We thank Drs. Colm Sweeney and Kirk Thoning, at the NOAA Global Monitoring Laboratory, for providing code and support for implementing the curve fitting methods. A dataset file provided via the Obspack from the Mauna Loa surface observing station is included in the tests directory with permission from the data provider, @KeelingEtAl2001sioref. This research was supported as part of the Energy Exascale Earth System Model (E3SM) project, funded by the U.S. Department of Energy (DOE), Office of Science, Office of Biological and Environmental Research. Data analysis described in this work relied on computational resources provided by the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract DE-AC02-05CH11231. The Pacific Northwest National Laboratory (PNNL) is operated for DOE by Battelle Memorial Institute under Contract DE-AC05-76RLO1830.

# Author contributions

D.K., K.C., B.H., and S.B. initially conceived the study. D.K. was the main code contributor of the gdess software and wrote the initial version of the paper. S.F. contributed to code testing. S.F., B.H., and S.B. ideated the experiment examples and priorities. All authors discussed the results, commented, and contributed to writing of the final version of the paper. K.C. supervised the study.

# References