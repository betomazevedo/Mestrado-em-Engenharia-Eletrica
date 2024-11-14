import pandas as pd
import numpy as np
import scipy.stats as sp
import torch

from dataset.dataset import MAEDataset

class RollingLabelStrategy:
    """
    Base class that just wraps applications of apply. 
    Leverages pandas' Rolling function.  
    (16/02/2024)DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=_NoDefault.no_default, closed=None, 
    step=None, method='single') Provide rolling window calculation
    Example:
    In [1]: s = pd.Series(range(5))
    In [2]: s.rolling(window=2).sum()
    Out[2]: 
    0    NaN
    1    1.0
    2    3.0
    3    5.0
    4    7.0
    dtype: float64

     
    (16022024)Rolling.apply(func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None) Calculate the rolling custom 
    aggregation function.
    Example:
    >>> ser = pd.Series([1, 6, 5, 4])
    >>> ser.rolling(2).apply(lambda s: s.sum() - s.min())
    0    NaN
    1    6.0
    2    6.0
    3    5.0
    dtype: float64
    
    (16/02/2024) labels.rolling(window=self.window_size).apply(f, raw=True) Provide rolling window calculations with window_size of 100 or 
    more, and apply according to "apply fuction" (according used classes above,BinaryMCLStrategy, MulticlassMCLStrategy, and 
    OVAMCLStrategy). 
    (16022024)For example, in MulticlassMCLStrategy class is the modal number, most common value, of number of slices created.
    
    * Constructor arguments:
        - **window_size: INT** -- Size of sliding window

        - **stride: INT** -- Number of samples between consecutive windows

        - **offset: INT** -- Control how much to offset each window

    * Methods:

        - **apply(y, event_type)**

        - **__call__(labels, event_type)**    

    Variables
        - labels -- Come from dataset.dataset.MAEDataset (line 184), and is the "Tag corresponding to instance label".
    """

    def __init__(self, window_size, stride=1, offset=0):
        self.window_size = window_size
        self.stride = stride
        self.offset = offset

    def apply(self, y, event_type):
        raise NotImplementedError

    def __call__(self, labels, event_type):
        def f(y):
            return self.apply(y, event_type)
    
        labels = labels.rolling(window=self.window_size).apply(f, raw=True)  
        return labels[self.offset :: self.stride]


class BinaryMCLStrategy(RollingLabelStrategy):
    """
    Window label gets assigned to most common value,
    mapping transients and faults of ALL classes to true
    """

    def apply(self, y, event_type=None):
        """
        Map all fault types to True and apply mode over window
        """
        return sp.mode(y > 0)[0] #Return an array of the modal (most common) value in the passed array (all windows different of 0)


class MulticlassMCLStrategy(RollingLabelStrategy):
    """
    Window label gets assigned to most common value,
    mapping transients and faults to the CORRESPONDING CLASS CODE
    """

    def apply(self, y, event_type=None):
        """
        Map transient codes to fault codes and apply mode over window
        """
        return sp.mode(y % 100)[0] #Return an array of the modal (most common) value in the passed array (number of windows, defined by the remainder of number of label by 100)


class OVAMCLStrategy(RollingLabelStrategy):
    """
    Window label gets assigned to most common value,
    mapping transients and faults of SPECIFIC CLASS to true
    """

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = fault_code

    def apply(self, y, event_type=None):
        return sp.mode(y % 100 == self.fault_code)[0]


class TorchLabelStrategy:   # Estudar!!!!
    """
    Base class that just wraps applications of apply.
    Leverages pytorch unfold function (see above, apply windowing).

    (16/02/2024) Tensor.unfold(dimension, size, step) → Tensor. Returns a view of the original tensor which contains all slices of size 
    "size" from self tensor in the dimension "dimension".  Step between two slices is given by step.  An additional dimension of size "size" 
    is appended in the returned tensor.  
    
    (16/02/2024) Parameters of Tensor.unfold(dimension, size, step):
        dimension (int) – dimension in which unfolding happens
        size (int) – the size of each slice that is unfolded
        step (int) – the step between each slice

    (16/02/2024) Example:
    >>> x = torch.arange(1., 8)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> x.unfold(0, 2, 1)
    tensor([[ 1.,  2.],
            [ 2.,  3.],
            [ 3.,  4.],
            [ 4.,  5.],
            [ 5.,  6.],
            [ 6.,  7.]])
    >>> x.unfold(0, 2, 2)
    tensor([[ 1.,  2.],
            [ 3.,  4.],
            [ 5.,  6.]])
    
    (16/02/2024) pd.Series provide a one-dimensional labeled array with window of 100 or more, and apply according to "apply function" 
    (according used classes above, TorchBinaryMCLStrategy, TorchBinaryMRLStrategy, TorchOVAMCLStrategy, TorchOVATransientMCLStrategy, 
    TorchOVAMRLStrategy, TorchMulticlassMCLStrategy, and TorchMulticlassMRLStrategy). 
        
    (16/02/2024)For example, in TorchMulticlassMRLStrategy returns conditionally: If window is NAN, results in NAN, otherwise returns the  remainder of the number of elements from previous windows divided by 100.

    Variables 
        - labels -- Come from dataset.dataset.MAEDataset (line 184), and is the "Tag corresponding to instance label".
    """

    def __init__(self, window_size, stride=1, offset=0):  #stride - Number of samples between consecutive windows
        self.window_size = window_size # defined in modules of experiments (in stat, from 100 to 1000)
        self.stride = stride
        self.offset = offset

    def apply(self, y, event_type):
        raise NotImplementedError

    def __call__(self, labels, event_type):
        # store index
        index = labels.index

        # not enough samples for windowing, return empty
        if len(labels) < self.offset + self.window_size:  # If quantity of labels of some fail is less than size of window_size:
            out = pd.Series(name=MAEDataset.LABEL_NAME, dtype=np.float64)  # Create serie with these labels
            out.index.name = index.name # Select the index of these series
            return out

        # pass to pytorch as float (propagate nan)
        labels = torch.tensor(labels.values, dtype=torch.float32).squeeze() # create a multidimensional matrix of labels, but remove unitary elements

        # apply windowing (slices of window_size)
        y = labels[self.offset :].unfold(0, self.window_size, self.stride)  # y = labels.unfold(0, window_size, 1)
        index = index[self.offset :][self.window_size - 1 :: self.stride] # to extract (window size - 1) elements of index
          
        out = pd.Series(    # Constructing Series from a dictionary with an Index specified "index"
            name=MAEDataset.LABEL_NAME,
            data=self.apply(y, event_type), # Choose the label according to "apply fuction" 
            index=index,
            dtype=np.float64,
        )
        out.index.name = index.name
        return out


class TorchBinaryMCLStrategy(TorchLabelStrategy):
    """
    Any fault indicator, most common label
    """

    def apply(self, y, event_type=None):
        return torch.mode(y, dim=-1)[0] > 0


class TorchBinaryMRLStrategy(TorchLabelStrategy):
    """
    Any fault indicator, most recent label
    """

    def apply(self, y, event_type=None):
        return y[:, -1] > 0


class TorchOVAMCLStrategy(TorchLabelStrategy):
    """
    Specific class indicator, most common label
    """

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = fault_code

    def apply(self, y, event_type=None):
        return (torch.mode(y % 100)[0] == self.fault_code).float()


class TorchOVATransientMCLStrategy(TorchLabelStrategy):
    """
    Transients of specific class, most common label
    """

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = 100 + fault_code

    def apply(self, y, event_type=None):
        return 1.0 * torch.mode(y == self.fault_code)[0]


class TorchOVAMRLStrategy(TorchLabelStrategy):
    """transients and faults of specific class, most recent label, propagates nans"""

    _NAN = torch.tensor([np.nan]).float()

    def __init__(self, fault_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_code = fault_code

    def apply(self, y, event_type=None):
        last = y[:, -1]
        return torch.where(
            last.isnan(), self._NAN, (last % 100 == self.fault_code).float()
        )


class TorchMulticlassMCLStrategy(TorchLabelStrategy):  #
    """detect transients and faults, most common value.
        (16/02/2024) torch.mode -> Returns a namedtuple (values, indices) where values is the mode (most common) value of each row of the input tensor in the 
    given dimension dim, i.e. a value which appears most often in that row, and indices is the index location of each mode value found. 
        By default, dim is the last dimension of the input tensor (1).
    """   

    def apply(self, y, event_type=None):
        return torch.mode(y % 100, dim=-1)[0] 


class TorchMulticlassMRLStrategy(TorchLabelStrategy):
    """detect transients and faults, most common value. (27/12/23) It must be most recent label!!!

    torch.where(condition, input, other, *, out=None) Return a tensor of elements selected from either input or other, depending on condition.
    Returns conditionally: If window is NAN, results in NAN, otherwise returns the remainder of the number of label
    divided by 100.  For example, if the label is 105,  return 5 as tensors element.
    """   

    _NAN = torch.tensor([np.nan]).float()

    def apply(self, y, event_type=None):
        last = y[:, -1]  # (stablished in line 178),tensor of last slice until tensor of first slice, and just last column (label)
        # remainder = last[-1] % 100    # 26/02/24 último label extraído
        # print("qty of labels's tensor of slices:",last.size(), "first label:", last[1], "last label:", last[-1],"last remainder",remainder)  # 26/02/24
        return torch.where(last.isnan(), self._NAN, (last % 100).float()) 

class TorchMulticlassMRLStrategy2(TorchLabelStrategy):
    """detect transients and faults,  most recent label
    Separate 0 class normal from 0 class pseudo_normal.
         """   
    
    _NAN = torch.tensor([np.nan]).float()
    
    def apply(self, y, event_type=None):
        
        last = y[:, -1]
        if last[1] == 0. and last[-1] >= 1.:
            last = torch.where(last == 0., 9.0, last.float()) 
        if last[1] == 0. and last[-1] == self._NAN:
            last = torch.where(last == 0., 9.0, last.float()) 
        
        return torch.where(last.isnan(), self._NAN, (last % 100).float()) 

# class TorchMulticlassMRLStrategy2(TorchLabelStrategy):
#     """detect transients and faults,  most recent label
#     Separate 0 class normal from 0 class pseudo_normal.
#          """   
    
#     _NAN = torch.tensor([np.nan]).float()
    
#     def apply(self, y, event_type=None):
        
#         last = y[:, -1]
#         if event_type !=0:
#             last = torch.where(last == 0., 9.0, last.float()) 
        
#         return torch.where(last.isnan(), self._NAN, (last % 100).float()) 


class TorchMulticlassTransientMRL(TorchLabelStrategy):
    """detect transients, most RECENT LABEL."""
    
    _NAN = torch.tensor([np.nan]).float()   
    
    def apply(self, y, event_type=None):
        last = y[:, -1]
        last = torch.where((last > 0) & (last <=100), self._NAN, last.float())
                        
        return torch.where(last.isnan(), self._NAN, (last % 100).float()) 

class TorchMulticlassMRLClusterStrategy(TorchLabelStrategy):
    """detect transients and faults,  most recent label
    Separate 1 class to cluster before training.
         """   
    
    _NAN = torch.tensor([np.nan]).float()
    
    def apply(self, y, event_type=None):
        
        last = y[:, -1]
        if event_type !=1:
            last = torch.where(last>=0, self._NAN, last.float()) 
        
        return torch.where(last.isnan(), self._NAN, (last % 100).float())