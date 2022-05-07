import torch
import numpy as np
from . import MultiOutputKernel, Parameter, config

class IndependentMultiOutputKernel(MultiOutputKernel):
    """
    Kernel with subkernels for each channels independently. Only the subkernels as block matrices on the diagonal are calculated, there is no correlation between channels.

    Args:
        kernels (list of Kernel): Kernels of shape (output_dims,).
        output_dims (int): Number of output dimensions.
        name (str): Kernel name.
    """
    def __init__(self, *kernels, output_dims=None, name="IMO"):
        if output_dims is None:
            output_dims = len(kernels)
        super().__init__(output_dims, name=name)
        self.kernels = self._check_kernels(kernels, output_dims)

    def __getitem__(self, key):
        return self.kernels[key]
    
    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if i == j:
            return self.kernels[i].K(X1, X2)
        else:
            if X2 is None:
                X2 = X1
            return torch.zeros(X1.shape[0], X2.shape[0], device=config.device, dtype=config.dtype)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.kernels[i].K_diag(X1)

class MultiOutputSpectralKernel(MultiOutputKernel):
    """
    Multi-output spectral kernel (MOSM) where each channel and cross-channel is modelled with a spectral kernel as proposed by [1]. You can add the mixture kernel with `MixtureKernel(MultiOutputSpectralKernel(...), Q=3)`.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="MOSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        # TODO: incorporate mixtures?
        # TODO: allow different input_dims per channel
        magnitude = torch.rand(output_dims)
        mean = torch.rand(output_dims, input_dims)
        variance = torch.rand(output_dims, input_dims)
        delay = torch.zeros(output_dims, input_dims)
        phase = torch.zeros(output_dims)

        self.input_dims = input_dims
        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        if 1 < output_dims:
            self.delay = Parameter(delay)
            self.phase = Parameter(phase)

        self.twopi = np.power(2.0*np.pi,float(self.input_dims)/2.0)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        if i == j:
            variance = self.variance()[i]
            alpha = self.magnitude()[i]**2 * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5*torch.tensordot(tau**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, self.mean()[i], dims=1))  # NxM
            return alpha * exp * cos
        else:
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D

            diff_mean = self.mean()[i] - self.mean()[j]  # D
            magnitude = self.magnitude()[i]*self.magnitude()[j]*torch.exp(-np.pi**2 * diff_mean.dot(inv_variances*diff_mean))  # scalar

            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            delay = self.delay()[i] - self.delay()[j]  # D
            phase = self.phase()[i] - self.phase()[j]  # scalar

            alpha = magnitude * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * (torch.tensordot(tau+delay, mean, dims=1) + phase))  # NxM
            return alpha * exp * cos

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        variance = self.variance()[i]
        alpha = self.magnitude()[i]**2 * self.twopi * variance.prod().sqrt()  # scalar
        return alpha.repeat(X1.shape[0])

class UncoupledMultiOutputSpectralKernel(MultiOutputKernel):
    """
    Uncoupled multi-output spectral kernel (uMOSM) where each channel and cross-channel is modelled with a spectral kernel. It is similar to the MOSM kernel but instead of training a magnitude per channel, we train the lower triangle of the magnitude between all channels. You can add the mixture kernel with `MixtureKernel(UncoupledMultiOutputSpectralKernel(...), Q=3)`.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="uMOSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        magnitude = torch.rand(output_dims, output_dims).tril()
        mean = torch.rand(output_dims, input_dims)
        variance = torch.rand(output_dims, input_dims)
        delay = torch.zeros(output_dims, input_dims)
        phase = torch.zeros(output_dims)

        self.input_dims = input_dims
        self.magnitude = Parameter(magnitude)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        if 1 < output_dims:
            self.delay = Parameter(delay)
            self.phase = Parameter(phase)

        self.twopi = np.power(2.0*np.pi,float(self.input_dims)/2.0)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        magnitude = self.magnitude().tril().mm(self.magnitude().tril().T)
        if i == j:
            variance = self.variance()[i]
            alpha = magnitude[i,i] * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5*torch.tensordot(tau**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, self.mean()[i], dims=1))  # NxM
            return alpha * exp * cos
        else:
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D

            diff_mean = self.mean()[i] - self.mean()[j]  # D
            magnitude = magnitude[i,j] * torch.exp(-np.pi**2 * diff_mean.dot(inv_variances*diff_mean))  # scalar

            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            delay = self.delay()[i] - self.delay()[j]  # D
            phase = self.phase()[i] - self.phase()[j]  # scalar

            alpha = magnitude * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau+delay, mean, dims=1) + phase)  # NxM
            return alpha * exp * cos

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        magnitude = self.magnitude().tril().mm(self.magnitude().tril().T)
        variance = self.variance()[i]
        alpha = magnitude[i,i] * self.twopi * variance.prod().sqrt()  # scalar
        return alpha.repeat(X1.shape[0])

class CrossSpectralKernel(MultiOutputKernel):
    """
    Cross Spectral kernel as proposed by [1]. You can add the mixture kernel with `MixtureKernel(CrossSpectralKernel(...), Q=3)`.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        Rq (int): Number of subcomponents.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    [1] K.R. Ulrich et al, "GP Kernels for Cross-Spectrum Analysis", Advances in Neural Information Processing Systems 28, 2015
    """
    def __init__(self, output_dims, input_dims=1, Rq=1, active_dims=None, name="CSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        amplitude = torch.rand(output_dims, Rq)
        mean = torch.rand(input_dims)
        variance = torch.rand(input_dims)
        shift = torch.zeros(output_dims, Rq)

        self.input_dims = input_dims
        self.Rq = Rq
        self.amplitude = Parameter(amplitude, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        self.shift = Parameter(shift)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        if i == j:
            # put Rq into third dimension and sum at the end
            amplitude = self.amplitude()[i].reshape(1,1,-1)  # 1x1xRq
            exp = torch.exp(-0.5 * torch.tensordot(tau**2, self.variance(), dims=1)).unsqueeze(2)  # NxMx1
            # the following cos is as written in the paper, instead we take phi out of the product with the mean
            #cos = torch.cos(torch.tensordot(tau.unsqueeze(2), self.mean(), dims=1))
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, self.mean(), dims=1).unsqueeze(2)) # NxMxRq
            return torch.sum(amplitude * exp * cos, dim=2)
        else:
            shift = self.shift()[i] - self.shift()[j]  # Rq

            # put Rq into third dimension and sum at the end
            amplitude = torch.sqrt(self.amplitude()[i]*self.amplitude()[j]).reshape(1,1,-1)  # 1x1xRq
            exp = torch.exp(-0.5 * torch.tensordot(tau**2, self.variance(), dims=1)).unsqueeze(2)  # NxMx1
            # the following cos is as written in the paper, instead we take phi out of the product with the mean
            #cos = torch.cos(torch.tensordot(tau.unsqueeze(2) + shift.reshape(1,1,-1,1), self.mean(), dims=1))
            cos = torch.cos(2.0*np.pi * (torch.tensordot(tau, self.mean(), dims=1).unsqueeze(2) + shift.reshape(1,1,-1))) # NxMxRq
            return torch.sum(amplitude * exp * cos, dim=2)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        amplitude = self.amplitude()[i].sum()
        return amplitude.repeat(X1.shape[0])

class LinearModelOfCoregionalizationKernel(MultiOutputKernel):
    """
    Linear model of coregionalization kernel (LMC) as proposed by [1].

    Args:
        kernels (list of Kernel): Kernels of shape (Q,).
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        Q (int): Number of components.
        Rq (int): Number of subcomponents.
        name (str): Kernel name.

    [1] P. Goovaerts, "Geostatistics for Natural Resource Evaluation", Oxford University Press, 1997
    """
    def __init__(self, *kernels, output_dims, input_dims=1, Q=None, Rq=1, name="LMC"):
        super().__init__(output_dims, input_dims, name=name)

        if Q is None:
            Q = len(kernels)
        weight = torch.rand(output_dims, Q, Rq)

        self.kernels = self._check_kernels(kernels, Q)
        self.weight = Parameter(weight, lower=config.positive_minimum)

    def __getitem__(self, key):
        return self.kernels[key]

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        weight = torch.sum(self.weight()[i] * self.weight()[j], dim=1)  # Q
        kernels = torch.stack([kernel.K(X1,X2) for kernel in self.kernels], dim=2)  # NxMxQ
        return torch.tensordot(kernels, weight, dims=1)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        weight = torch.sum(self.weight()[i]**2, dim=1)  # Q
        kernels = torch.stack([kernel.K_diag(X1) for kernel in self.kernels], dim=1)  # NxQ
        return torch.tensordot(kernels, weight, dims=1)

class GaussianConvolutionProcessKernel(MultiOutputKernel):
    """
    Gaussian convolution process kernel (CONV) as proposed by [1].

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    [1] M.A. Álvarez and N.D. Lawrence, "Sparse Convolved Multiple Output Gaussian Processes", Advances in Neural Information Processing Systems 21, 2009
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="CONV"):
        super().__init__(output_dims, input_dims, active_dims, name)

        weight = torch.rand(output_dims)
        variance = torch.rand(output_dims, input_dims)
        base_variance = torch.rand(input_dims)

        self.input_dims = input_dims
        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=0.0)
        self.base_variance = Parameter(base_variance, lower=config.positive_minimum)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.squared_distance(X1,X2)  # NxMxD

        # differences with the thesis from Parra is that it lacks a multiplication of 2*pi, lacks a minus in the exponencial function, and doesn't write the variance matrices as inverted
        if X2 is None:
            variances = 2.0*self.variance()[i] + self.base_variance()  # D
            weight = self.weight()[i]**2 * torch.sqrt(self.base_variance().prod()/variances.prod())  # scalar
            exp = torch.exp(-0.5 * torch.tensordot(tau, 1.0/variances, dims=1))  # NxM
            return weight * exp
        else:
            variances = self.variance()[i] + self.variance()[j] + self.base_variance()  # D
            weight_variance = torch.sqrt(self.base_variance().prod()/variances.prod())  # scalar
            weight = self.weight()[i] * self.weight()[j] * weight_variance  # scalar
            exp = torch.exp(-0.5 * torch.tensordot(tau, 1.0/variances, dims=1))  # NxM
            return weight * exp

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        variances = 2.0*self.variance()[i] + self.base_variance()  # D
        weight = self.weight()[i]**2 * torch.sqrt(self.base_variance().prod()/variances.prod())  # scalar
        return weight.repeat(X1.shape[0])

class MultiOutputHarmonizableSpectralKernel(MultiOutputKernel):
    """
    Multi-output harmonizable spectral kernel (MOHSM) where each channel and cross-channel is modelled with a spectral kernel as proposed by [1]. You can add the mixture kernel with `MixtureKernel(MultiOutputHarmonizableSpectralKernel(...), Q=3)`.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    [1] M. Altamirano, "Nonstationary Multi-Output Gaussian Processes via Harmonizable Spectral Mixtures, 2021
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="MOHSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        # TODO: incorporate mixtures?
        # TODO: allow different input_dims per channel
        magnitude = torch.rand(output_dims)
        mean = torch.rand(output_dims, input_dims)
        variance = torch.rand(output_dims, input_dims)
        lengthscale = torch.rand(output_dims)
        delay = torch.zeros(output_dims, input_dims)
        phase = torch.zeros(output_dims)
        center = torch.zeros(input_dims)

        self.input_dims = input_dims
        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)
        
        if 1 < output_dims:
            self.delay = Parameter(delay)
            self.phase = Parameter(phase)
            
        self.twopi = np.power(2.0*np.pi, float(self.input_dims))
        self.center = Parameter(center)
    
    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        avg = self.average(X1,X2)  # NxMxD
        
        if i == j:
            variance = self.variance()[i]
            lengthscale = self.lengthscale()[i]**2
            
            alpha = self.magnitude()[i]**2 * self.twopi * variance.prod().sqrt() * torch.pow(lengthscale.sqrt(), float(self.input_dims))  # scalar
            exp1 = torch.exp(-0.5 * torch.tensordot(tau**2, variance, dims=1))  # NxM
            exp2 = torch.exp(-0.5 * torch.tensordot((avg-self.center())**2, lengthscale*torch.ones(self.input_dims, device=config.device, dtype=config.dtype), dims=1))  # NxM
            cos = torch.cos(2.0 * np.pi * torch.tensordot(tau, self.mean()[i], dims=1))  # NxM
            return alpha * exp1 * cos * exp2
        else:
            lengthscale_i = self.lengthscale()[i]**2
            lengthscale_j = self.lengthscale()[j]**2
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D
            inv_lengthscale = 1.0/(lengthscale_i + lengthscale_j)  # D
            diff_mean = self.mean()[i] - self.mean()[j]  # D
            
            magnitude = self.magnitude()[i]*self.magnitude()[j] * torch.exp(-np.pi**2 * diff_mean.dot(inv_variances*diff_mean))  # scalar
            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            lengthscale = 2.0 * lengthscale_i * inv_lengthscale * lengthscale_j  # D
            delay = self.delay()[i] - self.delay()[j]  # D
            phase = self.phase()[i] - self.phase()[j]  # scalar

            alpha = magnitude * self.twopi * variance.prod().sqrt()*torch.pow(lengthscale.sqrt(),float(self.input_dims))  # scalar
            exp1 = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            exp2 = torch.exp(-0.5 * torch.tensordot((avg-self.center())**2, lengthscale*torch.ones(self.input_dims, device=config.device, dtype=config.dtype), dims=1))  # NxM
            cos = torch.cos(2.0 * np.pi * torch.tensordot(tau+delay, mean, dims=1) + phase)  # NxM
            return alpha * exp1 * cos * exp2

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        variance = self.variance()[i]
        lengthscale = self.lengthscale()[i]**2
        alpha = self.magnitude()[i]**2 * self.twopi * variance.prod().sqrt() * torch.pow(lengthscale.sqrt(), float(self.input_dims))  # scalar
        exp2 = torch.exp(-0.5 * torch.tensordot((X1-self.center())**2, lengthscale*torch.ones(self.input_dims, device=config.device, dtype=config.dtype), dims=1))  # NxM
        return alpha * exp2
