
def is_numeric(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def is_positive(x):
    if is_numeric(x):
        return float(x) > 0
    else:
        return False

def is_positive_or_zero(x):
    if is_numeric(x):
        return float(x) >= 0
    else:
        return False

# will intentionally accept floats with integer values
def is_integer(x):
    if is_numeric(x):
        return int(float(x)) == float(x)
    else:
        return False

# will intentionally accept 0, 1, "True", "False", "0", "1"
def is_bool(x):
    if x in [True, False, 0, 1, "True", "False", "0", "1"]:
        return True
    else:
        return False

def is_string(x):
    return isinstance(x, str)

def is_positive_integer(x):
    return is_numeric(x) and is_integer(x) and is_positive(x)

def is_positive_integer_or_zero(x):
    return is_numeric(x) and is_integer(x) and is_positive_or_zero(x)

def is_negative_integer(x):
    if is_integer(x):
        return not is_positive(x)
    else:
        return False

# Custom exception to raise when we intentinoally catch an error
# This way we can test that the right error was raised in test cases
class InputError(Exception):
    pass
    #def __init__(self, msg, loc):
    #    self.msg = msg
    #    self.loc = loc
    #def __str__(self):
    #    return repr(self.msg)
