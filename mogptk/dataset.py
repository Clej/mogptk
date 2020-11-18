import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import Data

def LoadCSV(filename, x_col=0, y_col=1, name=None, **kwargs):
    """
    LoadCSV loads a dataset from a given CSV file. It loads in x_cols as the names of the input dimension columns, and y_cols the name of the output columns. A filter can be set to filter out data from the CSV, such as ensuring that another column has a certain value.
    Args:
        filename (str): CSV filename.
        x_col (int, str, list of int or str): Names or indices of X column(s) in CSV.
        y_col (int, str, list of int or str): Names or indices of Y column(s) in CSV.
        name (str, list, optional): Name or names of data channels.
        **kwargs: Additional keyword arguments for csv.DictReader.

    Returns:
        mogptk.data.Data or mogptk.dataset.DataSet

    Examples:
        >>> LoadCSV('gold.csv', 'Date', 'Price', name='Gold')
        <mogptk.dataset.DataSet at ...>
        >>> LoadCSV('gold.csv', 'Date', 'Price', sep=' ', quotechar='|')
        <mogptk.dataset.DataSet at ...>
    """

    df = pd.read_csv(filename, **kwargs)

    return LoadDataFrame(df, x_col, y_col, name)

def LoadDataFrame(df, x_col=0, y_col=1, name=None):
    """
    LoadDataFrame loads a DataFrame from Pandas. It loads in x_cols as the names of the input dimension columns, and y_cols the names of the output columns.

    Args:
        df (pandas.DataFrame): The Pandas DataFrame.
        x_col (int, str, list of int or str): Names or indices of X column(s) in DataFrame.
        y_col (int, str, list of int or str): Names or indices of Y column(s) in DataFrame.
        name (str, list of str, optional): Name or names of data channels.

    Returns:
        mogptk.data.Data or mogptk.dataset.DataSet

    Examples:
        >>> df = pd.DataFrame(...)
        >>> LoadDataFrame(df, 'Date', 'Price', name='Gold')
        <mogptk.dataset.DataSet at ...>
    """

    if (not isinstance(x_col, list) or not all(isinstance(item, int) for item in x_col) and not all(isinstance(item, str) for item in x_col)) and not isinstance(x_col, int) and not isinstance(x_col, str):
        raise ValueError("x_col must be integer, string or list of integers or strings")
    if (not isinstance(y_col, list) or not all(isinstance(item, int) for item in y_col) and not all(isinstance(item, str) for item in y_col)) and not isinstance(y_col, int) and not isinstance(y_col, str):
        raise ValueError("y_col must be integer, string or list of integers or strings")

    if not isinstance(x_col, list):
        x_col = [x_col]
    if not isinstance(y_col, list):
        y_col = [y_col]

    if name is None:
        name = [None] * len(y_col)
    else:
        if not isinstance(name, list):
            name = [name]
        if len(y_col) != len(name):
            raise ValueError("y_col and name must be of the same length")

    # if columns are indices, convert to column names
    if all(isinstance(item, int) for item in x_col):
        x_col = [df.columns[item] for item in x_col]
    if all(isinstance(item, int) for item in y_col):
        y_col = [df.columns[item] for item in y_col]

    df = df[x_col + y_col]
    if len(df.index) == 0:
        raise ValueError("dataframe cannot be empty")

    input_dims = len(x_col)
    x_data = df[x_col]
    x_labels = [str(item) for item in x_col]

    dataset = DataSet()
    for i in range(len(y_col)):
        channel = df[x_col + [y_col[i]]].dropna()

        dataset.append(Data(
            channel[x_col].values,
            channel[y_col[i]].values,
            name=name[i],
            x_labels=x_labels,
            y_label=str(y_col[i]),
        ))
    if dataset.get_output_dims() == 1:
        return dataset[0]
    return dataset

################################################################
################################################################
################################################################

class DataSet:
    """
    DataSet is a class that holds multiple Data objects as channels.

    Args:
        *args (mogptk.data.Data, mogptk.dataset.DataSet, list, dict, np.ndarray): Accepts multiple arguments, each of which should be either a DataSet or Data object, a list of Data objects or a dictionary of Data objects. Each Data object will be added to the list of channels. In case of a dictionary, the key will set the name of the Data object. If a DataSet is passed, its channels will be added. It is also possible to pass x and y data array directly by either passing two np.ndarrays or two lists of np.ndarrays for x and y data.

    Examples:
        Three different ways to use DataSet:
        >>> wind_velocity = mogptk.LoadDataFrame(df, x_col='Date', y_col='Wind Velocity', name='wind')
        >>> tidal_height = mogptk.LoadDataFrame(df, x_col='Date', y_col='Tidal Height', name='tidal')
        >>> dataset = mogptk.DataSet(wind_velocity, tidal_height)

        >>> dataset = mogptk.DataSet(
        >>>     mogptk.LoadDataFrame(df, x_col='Date', y_col='Wind Velocity', name='wind'),
        >>>     mogptk.LoadDataFrame(df, x_col='Date', y_col='Tidal Height', name='tidal'),
        >>> )

        >>> dataset = mogptk.DataSet()
        >>> dataset.append(mogptk.LoadDataFrame(df, x_col='Date', y_col='Wind Velocity', name='wind'))
        >>> dataset.append(mogptk.LoadDataFrame(df, x_col='Date', y_col='Tidal Height', name='tidal'))

        >>> dataset = mogptk.DataSet(x, y)

        >>> dataset = mogptk.DataSet(x, [y1, y2, y3], names=['A', 'B', 'C'])
        
        >>> dataset = mogptk.DataSet([x1, x2, x3], [y1, y2, y3])

        Accessing individual channels:
        >>> dataset[0]       # first channel
        >>> dataset['wind']  # wind velocity channel
    """
    def __init__(self, *args, names=None):
        self.channels = []
        if len(args) == 2 and (isinstance(args[1], np.ndarray) or isinstance(args[1], list) and all(isinstance(item, np.ndarray) for item in args[1])):
            if names is None or isinstance(names, str):
                names = [names]

            if isinstance(args[0], np.ndarray):
                for name, y in zip(names, args[1]):
                    self.append(Data(args[0], y, name=name))
                return
            elif isinstance(args[0], list) and all(isinstance(item, np.ndarray) for item in args[0]) and isinstance(args[1], list) and len(args[0]) == len(args[1]):
                for name, x, y in zip(names, args[0], args[1]):
                    self.append(Data(x, y, name=name))
                return

        for arg in args:
            self.append(arg)

    def __iter__(self):
        return self.channels.__iter__()

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.channels[self.get_names().index(key)]
        return self.channels[key]

    def __setitem__(self, key, arg):
        if isinstance(arg, Data):
            self.channels[key] = arg
        elif isinstance(arg, DataSet) and len(arg) == 1:
            self.channels[key] = arg[0]
        else:
            raise Exception("must set a data type of Data or a DataSet with a single channel")

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ''
        for channel in self.channels:
            s += channel.__repr__() + "\n"
        return s

    def append(self, arg):
        """
        Append channel(s) to DataSet.
        
        Args:
            arg (mogptk.data.Data, mogptk.dataset.DataSet, list, dict): Argument can be either a DataSet or Data object, a list of Data objects or a dictionary of Data objects. Each Data object will be added to the list of channels. In case of a dictionary, the key will set the name of the Data object. If a DataSet is passed, its channels will be added.

        Examples:
            >>> dataset.append(mogptk.LoadFunction(lambda x: np.sin(5*x[:,0]), n=200, start=0.0, end=4.0, name='A'))
        """
        if isinstance(arg, Data):
            self.channels.append(arg)
        elif isinstance(arg, DataSet):
            for val in arg.channels:
                self.channels.append(val)
        elif isinstance(arg, list) and all(isinstance(val, Data) for val in arg):
            for val in arg:
                self.channels.append(val)
        elif isinstance(arg, dict) and all(isinstance(val, Data) for val in arg.values()):
            for key, val in arg.items():
                val.name = key
                self.channels.append(val)
        else:
            raise Exception("unknown data type %s in append to DataSet" % (type(arg)))
        return self

    def get_input_dims(self):
        """
        Return the input dimensions per channel.

        Returns:
            list: List of input dimensions per channel.

        Examples:
            >>> dataset.get_input_dims()
            [2, 1]
        """
        return [channel.get_input_dims() for channel in self.channels]

    def get_output_dims(self):
        """
        Return the output dimensions of the dataset, i.e. the number of channels.

        Returns:
            int: Output dimensions.

        Examples:
            >>> dataset.get_output_dims()
            4
        """
        return len(self.channels)

    def get_names(self):
        """
        Return the names of the channels.

        Returns:
            list: List of names.

        Examples:
            >>> dataset.get_names()
            ['A', 'B', 'C']
        """
        return [channel.get_name() for i, channel in enumerate(self.channels)]

    def get(self, index):
        """
        Return Data object given a channel index or name.

        Args:
            index (int, str): Index or name of the channel.

        Returns:
            mogptk.data.Data: Channel data.

        Examples:
            >>> channel = dataset.get('A')
        """
        if isinstance(index, int):
            if index < len(self.channels):
                return self.channels[index]
        elif isinstance(index, str):
            for channel in self.channels:
                if channel.name == index:
                    return channel
        raise ValueError("channel '%d' does not exist in DataSet" % (index))
    
    def get_index(self, index):
        """
        Return channel numeric index given a channel index or name.

        Args:
            index (int, str): Index or name of the channel.

        Returns:
            int: Channel index.

        Examples:
            >>> channel_index = dataset.get_index('A')
        """
        if isinstance(index, int):
            if index < len(self.channels):
                return index
        elif isinstance(index, str):
            for channel in self.channels:
                if channel.name == index:
                    return index
        raise ValueError("channel '%d' does not exist in DataSet" % (index))
    
    def get_data(self, transformed=False):
        """
        Returns all observations, train and test.

        Arguments:
            transformed (boolean, optional): Return transformed data.

        Returns:
            list: X data of shape (n,input_dims) per channel.
            list: Y data of shape (n,) per channel.

        Examples:
            >>> x, y = dataset.get_data()
        """
        return [channel.get_data(transformed=transformed)[0] for channel in self.channels], [channel.get_data(transformed=transformed)[1] for channel in self.channels]
    
    def get_train_data(self, transformed=False):
        """
        Returns observations used for training.

        Arguments:
            transformed (boolean, optional): Return transformed data.

        Returns:
            list: X data of shape (n,input_dims) per channel.
            list: Y data of shape (n,) per channel.

        Examples:
            >>> x, y = dataset.get_train_data()
        """
        return [channel.get_train_data(transformed=transformed)[0] for channel in self.channels], [channel.get_train_data(transformed=transformed)[1] for channel in self.channels]

    def get_test_data(self, transformed=False):
        """
        Returns the observations used for testing which correspond to the 
        removed points.

        Arguments:
            transformed (boolean, optional): Return transformed data.

        Returns:
            list: X data of shape (n,input_dims) per channel.
            list: Y data of shape (n,) per channel.

        Examples:
            >>> x, y = dataset.get_test_data()
        """
        return [channel.get_test_data(transformed=transformed)[0] for channel in self.channels], [channel.get_test_data(transformed=transformed)[1] for channel in self.channels]
    
    def get_prediction_x(self, transformed=False):
        """
        Returns the prediction X range for all channels.

        Args:
            transformed (boolean, optional): Return transformed data as used for training.

        Returns:
            list: X prediction of shape (n,input_dims) per channel.

        Examples:
            >>> x = dataset.get_prediction()
        """
        x = []
        for channel in self.channels:
            x.append(channel.get_prediction_x(transformed=transformed))
        return x
    
    def get_prediction(self, name, sigma=2.0, transformed=False):
        """
        Returns the prediction of a given name with a normal variance of sigma for all channels.

        Args:
            name (str): Name of the prediction, equals the name of the model that made the prediction.
            sigma (float, optional): The uncertainty interval's number of standard deviations.
            transformed (boolean, optional): Return transformed data as used for training.

        Returns:
            list: X prediction of shape (n,input_dims) per channel.
            list: Y mean prediction of shape (n,) per channel.
            list: Y lower prediction of uncertainty interval of shape (n,) per channel.
            list: Y upper prediction of uncertainty interval of shape (n,) per channel.

        Examples:
            >>> x, y_mean, y_var_lower, y_var_upper = dataset.get_prediction('MOSM', sigma=1)
        """
        x = []
        mu = []
        lower = []
        upper = []
        for channel in self.channels:
            cx, cmu, clower, cupper = channel.get_prediction(name, sigma, transformed=transformed)
            x.append(cx)
            mu.append(cmu)
            lower.append(clower)
            upper.append(cupper)
        return x, mu, lower, upper

    def set_prediction_x(self, x):
        """
        Set the prediction range per channel.

        Args:
            x (list, dict): Array of shape (n,) or (n,input_dims) per channel with prediction X values. If a dictionary is passed, the index is the channel index or name.

        Examples:
            >>> dataset.set_prediction_x([[5.0, 5.5, 6.0, 6.5, 7.0], [0.1, 0.2, 0.3]])
            >>> dataset.set_prediction_x({'A': [5.0, 5.5, 6.0, 6.5, 7.0], 'B': [0.1, 0.2, 0.3]})
        """
        if isinstance(x, list):
            if len(x) != len(self.channels):
                raise ValueError("prediction x expected to be a list of shape (output_dims,n)")

            for i, channel in enumerate(self.channels):
                channel.set_prediction_x(x[i])
        elif isinstance(x, dict):
            for name in x:
                self.get(name).set_prediction_x(x[name])
        else:
            for i, channel in enumerate(self.channels):
                channel.set_prediction_x(x)

    def set_prediction_range(self, start, end, n=None, step=None):
        """
        Set the prediction range per channel. Inputs should be lists of shape (input_dims,) for each channel or dicts where the keys are the channel indices.

        Args:
            start (list, dict): Start values for prediction range per channel.
            end (list, dict): End values for prediction range per channel.
            n (list, dict, optional): Number of points for prediction range per channel.
            step (list, dict, optional): Step size for prediction range per channel.

        Examples:
            >>> dataset.set_prediction_range([2, 3], [5, 6], [4, None], [None, 0.5])
            >>> dataset.set_prediction_range(0.0, 5.0, n=200) # the same for each channel
        """
        if not isinstance(start, (list, dict)):
            start = [start] * self.get_output_dims()
        elif isinstance(start, dict):
            start = [start[name] for name in self.get_names()]
        if not isinstance(end, (list, dict)):
            end = [end] * self.get_output_dims()
        elif isinstance(end, dict):
            end = [end[name] for name in self.get_names()]
        if n is None:
            n = [None] * self.get_output_dims()
        elif not isinstance(n, (list, dict)):
            n = [n] * self.get_output_dims()
        elif isinstance(n, dict):
            n = [n[name] for name in self.get_names()]
        if step is None:
            step = [None] * self.get_output_dims()
        elif not isinstance(step, (list, dict)):
            step = [step] * self.get_output_dims()
        elif isinstance(step, dict):
            step = [step[name] for name in self.get_names()]

        if len(start) != len(self.channels) or len(end) != len(self.channels) or len(n) != len(self.channels) or len(step) != len(self.channels):
            raise ValueError("start, end, n, and/or step must be lists of shape (output_dims,n)")

        for i, channel in enumerate(self.channels):
            channel.set_prediction_range(start[i], end[i], n[i], step[i])

    def clear_predictions(self):
        """
        Clear all saved predictions for all channels.
        """
        for i, channel in enumerate(self.channels):
            channel.clear_predictions()
    
    def get_nyquist_estimation(self):
        """
        Estimate nyquist frequency by taking 0.5/(minimum distance of points).

        Returns:
            list: Nyquist frequency array of shape (input_dims) per channel.

        Examples:
            >>> freqs = dataset.get_nyquist_estimation()
        """
        return [channel.get_nyquist_estimation() for channel in self.channels]
    
    def get_lombscargle_estimation(self, Q=1, n=10000):
        """
        Peaks estimation using Lomb Scargle.

        Args:
            Q (int): Number of peaks to find, defaults to 1.
            n (int): Number of points of the grid to evaluate frequencies, defaults to 10000.

        Returns:
            list: Amplitude array of shape (Q,input_dims) per channel.
            list: Frequency array of shape (Q,input_dims) per channel.
            list: Variance array of shape (Q,input_dims) per channel.

        Examples:
            >>> amplitudes, means, variances = dataset.get_lombscargle_estimation()
        """
        amplitudes = []
        means = []
        variances = []
        for channel in self.channels:
            channel_amplitudes, channel_means, channel_variances = channel.get_lombscargle_estimation(Q, n)
            amplitudes.append(channel_amplitudes)
            means.append(channel_means)
            variances.append(channel_variances)
        return amplitudes, means, variances
    
    def get_bnse_estimation(self, Q=1, n=1000):
        """
        Peaks estimation using BNSE (Bayesian Non-parametric Spectral Estimation).

        Args:
            Q (int): Number of peaks to find, defaults to 1.
            n (int): Number of points of the grid to evaluate frequencies, defaults to 1000.

        Returns:
            list: Amplitude array of shape (Q,input_dims) per channel.
            list: Frequency array of shape (Q,input_dims) per channel.
            list: Variance array of shape (Q,input_dims) per channel.

        Examples:
            >>> amplitudes, means, variances = dataset.get_bnse_estimation()
        """
        amplitudes = []
        means = []
        variances = []
        for channel in self.channels:
            channel_amplitudes, channel_means, channel_variances = channel.get_bnse_estimation(Q, n)
            amplitudes.append(channel_amplitudes)
            means.append(channel_means)
            variances.append(channel_variances)
        return amplitudes, means, variances
    
    def get_sm_estimation(self, Q=1, method='BNSE', optimizer='Adam', iters=100, params={}, plot=False):
        """
        Peaks estimation using the Spectral Mixture kernel.

        Args:
            Q (int): Number of peaks to find, defaults to 1.
            method (str, optional): Method of estimating SM kernels.
            optimizer (str, optional): Optimization method for SM kernels.
            iters (str, optional): Maximum iteration for SM kernels.
            params (object, optional): Additional parameters for PyTorch optimizer.
            plot (bool, optional): Show the PSD of the kernel after fitting.

        Returns:
            list: Amplitude array of shape (Q,input_dims) per channel.
            list: Frequency array of shape (Q,input_dims) per channel.
            list: Variance array of shape (Q,input_dims) per channel.

        Examples:
            >>> amplitudes, means, variances = dataset.get_sm_estimation()
        """
        amplitudes = []
        means = []
        variances = []
        for channel in self.channels:
            channel_amplitudes, channel_means, channel_variances = channel.get_sm_estimation(Q, method, optimizer, iters, params, plot)
            amplitudes.append(channel_amplitudes)
            means.append(channel_means)
            variances.append(channel_variances)
        return amplitudes, means, variances

    def transform(self, transformer):
        """
        Transform each channel by using one of the provided transformers, such as `TransformDetrend`, `TransformLinear`, `TransformLog`, `TransformNormalize`, `TransformWhiten`, ...

        Args:
            transformer (obj): Transformer object derived from TransformBase.

        Examples:
            >>> dataset.transform(mogptk.TransformDetrend(degree=2))        # remove polynomial trend
            >>> dataset.transform(mogptk.TransformLinear(slope=1, bias=2))  # remove linear trend
            >>> dataset.transform(mogptk.TransformLog)                      # log transform the data
            >>> dataset.transform(mogptk.TransformNormalize)                # transform to [-1,1]
            >>> dataset.transform(mogptk.TransformWhiten)                   # transform to mean=0, var=1
        """
        for channel in self.channels:
            channel.transform(transformer)

    def filter(self, start, end):
        """
        Filter the data range to be between start and end.

        Args:
            start (float, str): Start of interval.
            end (float, str): End of interval.

        Examples:
            >>> dataset.filter(3, 8)

            >>> dataset.filter('2016-01-15', '2016-06-15')
        """
        for channel in self.channels:
            channel.filter(start, end)

    def aggregate(self, duration, f=np.mean):
        """
        Aggregate the data by duration and apply a function to obtain a reduced dataset.

        For example, group daily data by week and take the mean.
        The duration can be set as a number which defined the intervals on the X axis,
        or by a string written in the duration format with:
        y=year, M=month, w=week, d=day, h=hour, m=minute, and s=second.
        For example, 3w1d means three weeks and one day, ie. 22 days, or 6M to mean six months.

        Args:
            duration (float, str): Duration along the X axis or as a string in the duration format.
            f (function, optional): Function to use to reduce data, by default uses np.mean.

        Examples:
            >>> dataset.aggregate(5)

            >>> dataset.aggregate('2w', f=np.sum)
        """
        for channel in self.channels:
            channel.aggregate(duration, f)

    def copy(self):
        """
        Make a deep copy of DataSet.

        Returns:
            mogptk.dataset.DataSet

        Examples:
            >>> other = dataset.copy()
        """
        return copy.deepcopy(self)

    def plot(self, pred=None, title=None, figsize=None, legend=True, transformed=False):
        """
        Plot each Data channel.

        Args:
            pred (str, optional): Specify model name to draw.
            title (str, optional): Set the title of the plot.
            figsize (tuple, optional): Set the figure size.
            legend (boolean, optional): Disable legend.
            transformed (boolean, optional): Display transformed Y data as used for training.

        Returns:
            matplotlib.figure.Figure: The figure.
            list of matplotlib.axes.Axes: List of axes.

        Examples:
            >>> fig, axes = dataset.plot(title='Title')
        """
        if figsize is None:
            figsize = (12, 3.0 * len(self.channels))

        h = figsize[1]
        fig, axes = plt.subplots(self.get_output_dims(), 1, figsize=figsize, squeeze=False, constrained_layout=True)

        legends = {}
        for channel in range(self.get_output_dims()):
            ax = self.channels[channel].plot(pred=pred, ax=axes[channel,0], transformed=transformed)
            legend = ax.get_legend()
            for text, handle in zip(legend.texts, legend.legendHandles):
                if text.get_text() == "Training Points":
                    handle = plt.Line2D([0], [0], ls='-', color='k', marker='.', ms=10, label='Training Points')
                legends[text.get_text()] = handle
            legend.remove()

        legend_rows = (len(legends)-1)/5 + 1
        if title is not None:
            fig.suptitle(title, y=(h+0.2+0.4*legend_rows)/h, fontsize=18)

        fig.legend(handles=legends.values(), loc="upper center", bbox_to_anchor=(0.5,(h-0.3+0.5*legend_rows)/h), ncol=5)
        return fig, axes

    def plot_spectrum(self, title=None, method='lombscargle', per=None, maxfreq=None, figsize=None, transformed=False):
        """
        Plot each Data channel spectrum.

        Args:
            title (str, optional): Set the title of the plot.
            method (list, str, optional): Set the method to get the spectrum such as 'lombscargle'.
            per (list, str, optional): Set the scale of the X axis depending on the formatter used, eg. per=5, per='day', or per='3d'.
            maxfreq (list, float, optional): Maximum frequency to plot, otherwise the Nyquist frequency is used.
            figsize (tuple, optional): Set the figure size.
            transformed (boolean, optional): Display transformed Y data as used for training.

        Returns:
            matplotlib.figure.Figure: The figure.
            list of matplotlib.axes.Axes: List of axes.

        Examples:
            >>> fig, axes = dataset.plot_spectrum(title='Title', method='bnse')
        """
        if not isinstance(method, list):
            method = [method] * len(self.channels)
        if not isinstance(per, list):
            per = [per] * len(self.channels)
        if not isinstance(maxfreq, list):
            maxfreq = [maxfreq] * len(self.channels)

        if figsize is None:
            figsize = (12, 3.0 * len(self.channels))

        fig, axes = plt.subplots(self.get_output_dims(), 1, figsize=figsize, squeeze=False, constrained_layout=True)
        if title != None:
            fig.suptitle(title, fontsize=18)

        for channel in range(self.get_output_dims()):
            ax = self.channels[channel].plot_spectrum(method=method[channel], ax=axes[channel,0], per=per[channel], maxfreq=maxfreq[channel], transformed=transformed)
        return fig, axes
