import numpy as np

from ..dataset import DataSet
from ..model import Model, Exact, logger
from ..gpr import SpectralKernel, IndependentMultiOutputKernel, MixtureKernel, GaussianLikelihood
from ..plot import plot_spectrum

class SM(Model):
    """
    Independent Spectral Mixture kernels per channel. The spectral mixture kernel is proposed by [1]. The parameters will be randomly instantiated, use `init_parameters()` to initialize the parameters to reasonable values for the current data set.

    Args:
        dataset (mogptk.dataset.DataSet): `DataSet` object of data for all channels.
        Q (int): Number of components.
        model: Gaussian process model to use, such as `mogptk.model.Exact`.
        mean (mogptk.gpr.mean.Mean): The mean class.
        name (str): Name of the model.
        rescale_x (bool): Rescale the X axis to [0,1000] to help training.

    Attributes:
        dataset: The associated mogptk.dataset.DataSet.
        model: The mogptk.gpr.model.Model.
        kernel: The mogptk.gpr.kernel.Kernel.

    Examples:

    >>> import numpy as np
    >>> import mogptk
    >>> 
    >>> t = np.linspace(0, 10, 100)
    >>> y = np.sin(0.5 * t)
    >>> 
    >>> data = mogptk.Data(t, y)
    >>> model = mogptk.SM(data, Q=1)
    >>> model.init_parameters()
    >>> model.train()
    >>> model.predict()
    >>> data.plot()

    [1] A.G. Wilson and R.P. Adams, "Gaussian Process Kernels for Pattern Discovery and Extrapolation", International Conference on Machine Learning 30, 2013
    """
    def __init__(self, dataset, Q=1, inference=Exact(), mean=None, name="SM", rescale_x=False):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)

        kernel = IndependentMultiOutputKernel(
            [MixtureKernel(SpectralKernel(dataset[j].get_input_dims()), Q)
                for j in range(dataset.get_output_dims())],
            output_dims=dataset.get_output_dims(),
        )
        super().__init__(dataset, kernel, inference, mean, name, rescale_x)

        self.Q = Q
        nyquist = np.array(self.dataset.get_nyquist_estimation())
        for j in range(self.dataset.get_output_dims()):
            for q in range(Q):
                self.gpr.kernel[j][q].mean.assign(upper=nyquist[j])

    def init_parameters(self, method='LS'):
        """
        Estimate kernel parameters from the data set. The initialization can be done using three methods:

        - BNSE estimates the PSD via Bayesian non-parametris spectral estimation (Tobar 2018) and then selecting the greater Q peaks in the estimated spectrum, and use the peak's position, magnitude and width to initialize the mean, magnitude and variance of the kernel respectively.
        - LS is similar to BNSE but uses Lomb-Scargle to estimate the spectrum, which is much faster but may give poorer results.
        - IPS uses independent parameter sampling from the PhD thesis of Andrew Wilson 2014. It takes the inverse of the lengthscales drawn from a truncated Gaussian Normal(0, max_dist^2), the means drawn from a Unif(0, 0.5 / minimum distance between two points), and the mixture weights by taking the standard variation of the Y values divided by the number of mixtures.

        In all cases the noise is initialized with 1/30 of the variance of each channel.

        Args:
            method (str): Method of estimation, such as IPS, LS, or BNSE.
        """

        input_dims = self.dataset.get_input_dims()
        output_dims = self.dataset.get_output_dims()

        if method.lower() not in ['ips', 'ls', 'bnse']:
            raise ValueError("valid methods of estimation are IPS, LS, and BNSE")

        if method.lower() == 'ips':
            for j in range(output_dims):
                nyquist = self.dataset[j].get_nyquist_estimation()
                x, y = np.array([x.transformed[self.dataset[j].mask] for x in self.dataset[j].X]).T, self.dataset[j].Y.transformed[self.dataset[j].mask]
                x_range = np.max(x, axis=0) - np.min(x, axis=0)

                weights = [2.0*y.std()/self.Q] * self.Q
                means = nyquist * np.random.rand(self.Q, input_dims[j])
                variances = 1.0 / (np.abs(np.random.randn(self.Q, input_dims[j])) * x_range)

                for q in range(self.Q):
                    self.gpr.kernel[j][q].sigma.assign(np.sqrt(weights[q]))
                    self.gpr.kernel[j][q].mean.assign(means[q,:])
                    self.gpr.kernel[j][q].variance.assign(variances[q,:])
            return
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset.get_lombscargle_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('LS could not find peaks for SM')
                return
        elif method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
            if np.sum(amplitudes) == 0.0:
                logger.warning('BNSE could not find peaks for SM')
                return

        #if noise:
        #    # TODO: remove this and set noise parameter to 1/30
        #    pct = 1/30.0
        #    # noise proportional to the values
        #    noise_amp = np.random.multivariate_normal(
        #        mean=np.zeros(self.Q),
        #        cov=np.diag(amplitudes.mean(axis=1) * pct))
        #    # set value to a minimun value
        #    amplitudes = np.maximum(np.zeros_like(amplitudes) + 1e-6, amplitudes + noise_amp)

        #    noise_mean = np.random.multivariate_normal(
        #        mean=np.zeros(self.Q),
        #        cov=np.diag(means.mean(axis=1) * pct))
        #    means = np.maximum(np.zeros_like(means) + 1e-6, means + noise_mean)

        #    noise_var = np.random.multivariate_normal(
        #        mean=np.zeros(self.Q),
        #        cov=np.diag(variances.mean(axis=1) * pct))
        #    variances = np.maximum(np.zeros_like(variances) + 1e-6, variances + noise_var)

        for j in range(output_dims):
            # TODO: check weights for all kernels
            _, y = self.dataset[j].get_train_data(transformed=True)
            mixture_weights = amplitudes[j].mean(axis=1)
            if 0.0 < amplitudes[j].sum():
                mixture_weights /= amplitudes[j].sum()
            mixture_weights *= 2.0 * y.std()
            for q in range(self.Q):
                self.gpr.kernel[j][q].sigma.assign(np.sqrt(mixture_weights[q]))
                self.gpr.kernel[j][q].mean.assign(means[j][q,:])
                self.gpr.kernel[j][q].variance.assign(variances[j][q,:])

    def plot_spectrum(self, log=False, noise=False, title=None):
        """
        Plot spectrum of kernel.
        """
        names = self.dataset.get_names()
        nyquist = self.dataset.get_nyquist_estimation()
        output_dims = self.dataset.get_output_dims()

        means = np.array([[self.gpr.kernel[j][q].mean.numpy() for j in range(output_dims)] for q in range(self.Q)])
        scales = np.array([[np.sqrt(self.gpr.kernel[j][q].variance.numpy()) for j in range(output_dims)] for q in range(self.Q)])
        weights = np.array([[self.gpr.kernel[j][q].sigma.numpy()[0]**2 for j in range(output_dims)] for q in range(self.Q)])

        noises = None
        if noise:
            if not isinstance(self.gpr.likelihood, GaussianLikelihood):
                raise ValueError("likelihood must be Gaussian to enable spectral noise")
            if self.gpr.likelihood.variance().ndim == 2:
                raise ValueError("likelihood variance must not be per data point to enable spectral noise")
            noises = self.gpr.likelihood.variance.numpy()

        return plot_spectrum(means, scales, dataset=self.dataset, weights=weights, nyquist=nyquist, noises=noises, log=log, titles=names, title=title)
