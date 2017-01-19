try:
#    print('try')
    from numpy import *
    import numpy as np
except:
#    print('except')
    from numpypy import *
    import numpypy as np
#    print('quicksort')

def diff(a, n=1, axis=-1):
    """
    Calculate the n-th order discrete difference along given axis.

    The first order difference is given by ``out[n] = a[n+1] - a[n]`` along
    the given axis, higher order differences are calculated by using `diff`
    recursively.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.

    Returns
    -------
    diff : ndarray
        The `n` order differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`.

    See Also
    --------
    gradient, ediff1d

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.diff(x)
    array([ 1,  2,  3, -7])
    >>> np.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> np.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> np.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
                "order must be non-negative but got " + repr(n))
    a = asanyarray(a)
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return diff(a[slice1]-a[slice2], n-1, axis=axis)
    else:
        return a[slice1]-a[slice2]

def linspace(start, stop, num=50, endpoint=True, retstep=False):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop` ].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float (only if `retstep` is True)
        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).
    logspace : Samples uniformly distributed in log space.

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
        array([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
        (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    num = int(num)
    if num <= 0:
        return array([], float)
    if endpoint:
        if num == 1:
            return array([float(start)])
        step = (stop-start)/float((num-1))
        y = arange(0, num) * step + start
        y[-1] = stop
    else:
        step = (stop-start)/float(num)
        y = arange(0, num) * step + start
    if retstep:
        return y, step
    else:
        return y

def isscalar(num):
    """
    Returns True if the type of `num` is a scalar type.

    Parameters
    ----------
    num : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if `num` is a scalar type, False if it is not.

    Examples
    --------
    >>> np.isscalar(3.1)
    True
    >>> np.isscalar([3.1])
    False
    >>> np.isscalar(False)
    True

    """
    if isinstance(num, generic):
        return True
    else:
        #This is a hack due to numpypy's incompleteness. I only ever use those scalar data types.
        return type(num) in [int, float, np.int32, np.float64]

def iterable(y):
    """
    Check whether or not an object can be iterated over.

    Parameters
    ----------
    y : object
      Input object.

    Returns
    -------
    b : {0, 1}
      Return 1 if the object has an iterator method or is a sequence,
      and 0 otherwise.


    Examples
    --------
    >>> np.iterable([1, 2, 3])
    1
    >>> np.iterable(2)
    0

    """
    try: iter(y)
    except: return 0
    return 1


def histogram(a, bins = 10, range = None):
#    print(a)
    a = np.asarray(a)
    #copy-paste numpy bin creation
    if not iterable(bins):
        if isscalar(bins) and bins < 1:
            raise ValueError("`bins` should be a positive integer.")
        if range is None:
            if a.size == 0:
                # handle empty arrays. Can't determine range, so use 0-1.
                range = (0, 1)
            else:
                range = (a.min(), a.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = linspace(mn, mx, bins+1, endpoint=True)
    else:
        bins = asarray(bins)
        if (diff(bins) < 0).any():
            raise AttributeError(
                    'bins must increase monotonically.')
    binWidth = bins[1]-bins[0]
    n = np.zeros((bins.shape[0]-1), dtype=np.float)
    minVal = bins[0]
    for x in a:
        index = int(((x - minVal) / binWidth))
        if index < n.shape[0]:
            n[index] += 1
        else:
            n[-1] += 1
    return n, bins

    