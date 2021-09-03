import math


def numstr(number, decimalpoints: int) -> str:
    """ Print big numbers nicely.
    Add commas, and restrict decimal places

    Parameters
    ----------
    number : numeric
    decimalpoints : int
        Number of decimal points to which the output string is restricted

    Returns
    -------
    str
        nicely formatted number
    """
    fmtstr = '{:,.%sf}' % str(decimalpoints)
    return fmtstr.format(number)


def my_round(x, nearest: int = 10, direction: str = 'up'
             ) -> int:
    """Round to the nearest specified whole number

    Parameters
    ----------
    x : numeric
    nearest : int, default 10
    direction : str, default 'up'

    Returns
    -------
    int
    """
    # rounding method
    if direction == 'up':
        return math.ceil(math.ceil(x / float(nearest))) * nearest
    elif direction == 'down':
        return math.floor(math.floor(x / float(nearest))) * nearest
    else:
        raise ValueError("Unexpected direction given. Should be either 'up' or 'down'.")
