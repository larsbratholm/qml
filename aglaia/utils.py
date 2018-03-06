import numpy as np

def is_positive(x):
    return (is_positive_or_zero(x) and x != 0)

def is_positive_or_zero(x):
    return (not is_array_like(x) and _is_numeric(x) and x >= 0)

def is_array_like(x):
    return isinstance(x, (list, np.ndarray))

def is_positive_integer(x):
    return (is_positive_integer_or_zero(x) and x != 0)

def is_string(x):
    return isinstance(x, str)

def is_positive_integer_or_zero(x):
    return (not is_array_like(x) and _is_integer and x >= 0)

def is_none(x):
    return isinstance(x, type(None))

def is_dict(x):
    return isinstance(x, dict)

def _is_numeric(x):
    return isinstance(x, (float, int))

def is_numeric_array(x):
    if is_array_like(x):
        try:
            np.asarray(x, dtype=float)
            return True
        except (ValueError, TypeError):
            return False
    return False
    
def _is_integer(x):
    return isinstance(x, int)

# will intentionally accept 0, 1 as well
def is_bool(x):
    return (x in (True, False))

def is_non_zero_integer(x):
    return (_is_integer(x) and x != 0)

def is_positive_integer_or_zero_array(x):
    if is_array_like(x):
        try:
            if np.asarray(x, dtype=float) == np.asarray(x, dtype=int):
                return True
        except (ValueError, TypeError):
            pass
    return False

#
#def _is_numeric_array(x):
#    try:
#        arr = np.asarray(x, dtype = float)
#        return True
#    except (ValueError, TypeError):
#        return False
#
#def _is_numeric_scalar(x):
#    try:
#        float(x)
#        return True
#    except (ValueError, TypeError):
#        return False
#
#def is_positive(x):
#    if is_array(x) and _is_numeric_array(x):
#        return _is_positive_scalar(x)
#
#def _is_positive_scalar(x):
#    return float(x) > 0
#
#def _is_positive_array(x):
#    return np.asarray(x, dtype = float) > 0
#
#def is_positive_or_zero(x):
#    if is_numeric(x):
#        if is_array(x):
#            return is_positive_or_zero_array(x)
#        else:
#            return is_positive_or_zero_scalar(x)
#    else:
#        return False
#
#def is_positive_or_zero_array(x):
#
#
#def is_positive_or_zero_scalar(x):
#    return float(x) >= 0
#
#def is_integer(x):
#    if is_array(x)
#        return is_integer_array(x)
#    else:
#        return is_integer_scalar(x)
#
## will intentionally accept floats with integer values
#def is_integer_array(x):
#    if is_numeric(x):
#        return (np.asarray(x) == np.asarray(y)).all()
#    else:
#        return False
#
## will intentionally accept floats with integer values
#def is_integer_scalar(x):
#    if is_numeric(x):
#        return int(float(x)) == float(x)
#    else:
#        return False
#
#
#def is_string(x):
#    return isinstance(x, str)
#
#def is_positive_integer(x):
#    return (is_numeric(x) and is_integer(x) and is_positive(x))
#
#def is_positive_integer_or_zero(x):
#    return (is_numeric(x) and is_integer(x) and is_positive_or_zero(x))
#
#def is_negative_integer(x):
#    if is_integer(x):
#        return not is_positive(x)
#    else:
#        return False
#
#def is_non_zero_integer(x):
#    return (is_positive_integer(x) or is_negative_integer(x))


# Custom exception to raise when we intentinoally catch an error
# This way we can test that the right error was raised in test cases
class InputError(Exception):
    pass
    #def __init__(self, msg, loc):
    #    self.msg = msg
    #    self.loc = loc
    #def __str__(self):
    #    return repr(self.msg)

def ceil(a, b):
    """
    Returns a/b rounded up to nearest integer.

    """
    return -(-a//b)
