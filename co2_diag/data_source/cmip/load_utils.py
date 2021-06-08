# -- Define valid model choices --
model_choices = ['CMIP.CNRM-CERFACS.CNRM-ESM2-1.esm-hist.Amon.gr', 'CMIP.NCAR.CESM2.esm-hist.Amon.gn',
                 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn', 'CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1']


def model_substring(s: str) -> str:
    """Function used to allow specification of model names by only supplying a partial string match

    Example
    -------
    >>> model_substring('BCC')
    returns 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn'
    """
    options = [c for c in model_choices if s in c]
    if len(options) == 1:
        return options[0]
    return s