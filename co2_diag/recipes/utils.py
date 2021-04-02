
def get_recipe_param(cls, param_dict, param_key: str, default_value=None):
    """Validate a parameter in the parameter dictionary, and return default if it is not in the dictionary.

    Parameters
    ----------
    param_dict
    param_key
    default_value

    Returns
    -------
    The value from the dictionary, which can be of any type
    """
    value = default_value
    if param_dict and (param_key in param_dict):
        value = param_dict[param_key]
    return value
