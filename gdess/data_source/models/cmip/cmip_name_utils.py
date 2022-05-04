from gdess import load_config_file
from gdess.formatters.args import nullable_str
import os, re, shlex

# -- Define valid model choices --
# Get the default model choices from the config file. Split (on commas) the retrieved string into a list.
config = load_config_file()
config_model_values = config.get('CMIP', 'model_choices', vars=os.environ)
my_splitter = shlex.shlex(config_model_values, posix=True)
my_splitter.whitespace += ','
my_splitter.whitespace_split = True
cmip_model_choices = list(my_splitter)

full_model_name_pattern = re.compile(
        r'(?P<activityid>[a-zA-Z\d\-]+)\.(?P<institutionid>[a-zA-Z\d\-]+)\.'
        r'(?P<sourceid>[a-zA-Z\d\-]+)\.(?P<experimentid>[a-zA-Z\d\-]+)\.'
        r'(?P<tableid>[a-zA-Z\d\-]+)\.(?P<gridlabel>[a-zA-Z\d\-]+)')


def model_name_dict_from_valid_form(s: str) -> dict:
    """Transform model_name into a dictionary with the parts

    Parameters
    ----------
    s : str

    Raises
    ------
    ValueError, if the form of the input string does not match either form (1) or (2)
    """
    # The supplied string is expected to be either in a shortened form <source>.<experiment> or a full name.
    short_pattern = re.compile(
        r'(?P<sourceid>[a-zA-Z\d\-]+)\.(?P<experimentid>[a-zA-Z\d\-]+)')

    if match := full_model_name_pattern.search(s):
        return match.groupdict()
    elif match := short_pattern.search(s):
        return match.groupdict()
    else:
        raise ValueError("Expected at least a source_id with an experiment_id, in the form "
                         "<source_id>.<experiment_id>, e.g. 'BCC.esm-hist'. Got <%s>" % s)


def matched_model_and_experiment(s: str) -> str:
    """Function used to allow specification of model names by only supplying a partial string match

    This function first checks whether the input is a string and of the form:
        (1) source_id.experiment_id
        or
        (2) activity_id.institution_id.source_id.experiment_id.table_id.grid_label
    A full name (i.e., in form (2)) will be returned, if the input matches one of the defined model choices.
    If the input does not match a defined model choice, then the input string will be returned unchanged.

    Example
    -------
    >>> matched_model_and_experiment('BCC.esm-hist')
    returns 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn'
    """
    # Transform the full names of the model choices into a dictionary of source and experiment ids.
    valid = [full_model_name_pattern.search(m).groupdict() for m in cmip_model_choices]
    valid_source_names = [v['sourceid'] for v in valid]

    # The supplied string is expected to be either in a shortened form <source>.<experiment> or a full name.
    if nullable_str(s):
        supplied = model_name_dict_from_valid_form(s)
    else:
        return s

    # match the substring to one of the full model names
    options = [(i, c) for i, c in enumerate(valid_source_names)
               if supplied['sourceid'] in c]
    if len(options) == 1:
        if valid[options[0][0]]['experimentid'] == supplied['experimentid']:
            return cmip_model_choices[options[0][0]]
    return s
