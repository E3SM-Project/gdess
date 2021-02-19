import math


def numstr(number, decimalpoints: int) -> str:
    """ Print big numbers nicely.
    Add commas, and restrict decimal places
    """
    fmtstr = '{:,.%sf}' % str(decimalpoints)
    return fmtstr.format(number)


def my_round(x, nearest: int = 10, direction: str = 'up'
             ) -> int:
    """Round to the nearest specified whole number

    Parameters
    ----------
    x
    nearest
    direction

    Returns
    -------

    """
    # rounding method
    if direction == 'up':
        retval = math.ceil(math.ceil(x / float(nearest))) * nearest
    elif direction == 'down':
        retval = math.floor(math.floor(x / float(nearest))) * nearest
    else:
        raise ValueError
    return retval
