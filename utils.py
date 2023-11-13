import numpy as np
from .stolen_utils import crossmatch

def ordered_crossmatch(x, y, skip_bounds_checking=False):
    inds1, inds2 = crossmatch(x,y, skip_bounds_checking=skip_bounds_checking)
    order = np.argsort(inds1)
    inds1 = inds1[order]
    inds2 = inds2[order]
    return inds1, inds2

def associated_crossmatch(x, y1, y2, skip_bounds_checking=False):
    """
    Function for using crossmatch when a second value associated with the crossmatched y1 is desired.
    e.g. you have the tyipcal x, y (what we call y1) as defined in crossmatch where crossmatch is used as follows:
        Finds where the elements of ``x`` appear in the array ``y``, including repeats.

        The elements in x may be repeated, but the elements in y must be unique.
        The arrays x and y may be only partially overlapping.
    In addition to this step, there is a second y2 associated with y1 such that each y1 is associated to exactly one y2.
    This function is simply a convenient way to use the first crossmatch to access the proper entries of y2 associated with y1.
    
    y2 may be arbitrary dimensioned, so long as its first dimension equals the length of y2
    """
    inds1, inds2 = crossmatch(x,y1, skip_bounds_checking=skip_bounds_checking)
    # Preserve original order
    order = np.argsort(inds1)
    inds1 = inds1[order]
    inds2 = inds2[order]
    return y2[inds2]

def internal_join_crossmatch(xout, xin, yin, yout, return_inds=False, skip_bounds_checking=False):
    """
    Use crossmatch to link entries of xin and yin exactly as crossmatch does normally, then order the resultings indsx, indsy 
    to preserve order when inserting. Use these connections to set xout[inds1] = yout[inds2].
    This function is useful when you want to alter xout with yout through relationships between xin and yin.
    In this case, each xout corresponds to one xin, and each yout corresponds to one yin, so the crossmatch
    between xin and yin is the link.
    
    yin must be 1d unique to work with crossmatch but xout and yout can be arbitrary dimensioned as long as
    the first dimension equals len(yin)
    """
    
    # Standard crossmatch
    inds1, inds2 = crossmatch(xin, yin, skip_bounds_checking=skip_bounds_checking)
    
    # Preserve original order
    order = np.argsort(inds1)
    inds1 = inds1[order]
    inds2 = inds2[order]
    
    copy_xout = np.array(xout)
    copy_xout[inds1] = yout[inds2]
    
    if return_inds:
        return inds1, inds2, copy_xout[inds1]
    return copy_xout

def numerify(x, make_hashable=False, data_to_int=None):
    """
    Given a list of non_numeric values, return a list of numeric values such that each item in the original list
    corresponds to a number in the new list. Repeated items will be given the same number.
    """
    if data_to_int is None:
        data_to_int = {}
    num_x = []
    for item in x:
        entry = item
        if make_hashable:
            entry = tuple(item)
        if entry not in data_to_int:
            data_to_int[entry] = len(data_to_int)
        num_x.append(data_to_int[entry])
    return np.array(num_x), data_to_int
        
def non_numeric_crossmatch(x, y, skip_counds_checking=False, numerify_values=True, make_hashable=True):
    
    if numerify_values:
        num_x, int_dict = numerify(x, make_hashable=make_hashable )
        num_y, _ = numerify(y, make_hashable=make_hashable, data_to_int=int_dict )

    return crossmatch(num_x, num_y, skip_bounds_checking=skip_counds_checking)