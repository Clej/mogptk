import torch
import numpy as np
from . import config, Parameter

def identity(x):
    """
    The identity link function given by

    $$ y = x $$
    """
    return x

def square(x):
    """
    The square link function given by

    $$ y = x^2 $$
    """
    return torch.square(x)

def exp(x):
    """
    The exponential link function given by

    $$ y = e^{x} $$
    """
    return torch.exp(x)

def inv_probit(x):
    """
    The inverse probit link function given by

    $$ y = \\frac{1}{2} \\left(1 + \\operatorname{erf}(x/\\sqrt{2})\\right) $$
    """
    jitter = 1e-3
    return 0.5*(1.0+torch.erf(x/np.sqrt(2.0))) * (1.0-2.0*jitter) + jitter

# also inv_logit or sigmoid
def logistic(x):
    """
    The logistic, inverse logit, or sigmoid link function given by

    $$ y = \\frac{1}{1 + e^{-x}} $$
    """
    return 1.0/(1.0+torch.exp(-x))

class GaussHermiteQuadrature:
    def __init__(self, deg=20, t_scale=None, w_scale=None):
        t, w = np.polynomial.hermite.hermgauss(deg)
        t = t.reshape(-1,1)
        w = w.reshape(-1,1)
        if t_scale is not None:
            t *= t_scale
        if w_scale is not None:
            w *= w_scale
        self.t = torch.tensor(t, device=config.device, dtype=config.dtype)  # degx1
        self.w = torch.tensor(w, device=config.device, dtype=config.dtype)  # degx1
        self.deg = deg

    def __call__(self, mu, var, F):
        return F(mu + var.sqrt().mm(self.t.T)).mm(self.w)  # Nx1

class Likelihood(torch.nn.Module):
    """
    Base likelihood.

    Args:
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, quadratures=20):
        super().__init__()

        self.quadrature = GaussHermiteQuadrature(deg=quadratures, t_scale=np.sqrt(2), w_scale=1.0/np.sqrt(np.pi))
        self.output_dims = None

    def name(self):
        return self.__class__.__name__

    def __setattr__(self, name, val):
        if name == 'train':
            for p in self.parameters():
                p.train = val
            return
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val._name is None:
            val._name = '%s.%s' % (self.__class__.__name__, name)
        elif isinstance(val, torch.nn.ModuleList):
            for i, item in enumerate(val):
                for p in item.parameters():
                    p._name = '%s[%d].%s' % (self.__class__.__name__, i, p._name)
        super().__setattr__(name, val)

    def validate_y(self, y, X=None):
        """
        Validate whether the y input is within the likelihood's support.
        """
        pass

    def log_prob(self, y, f, X=None):
        """
        Calculate the log probability

        $$ \\log(p(y|f)) $$

        with \\(p(y|f)\\) our likelihood.
        """
        # y: Nx1
        # f: NxM
        raise NotImplementedError()

    def variational_expectation(self, y, mu, var, X=None):
        """
        Calculate the variational expectation

        $$ \\int \\log(p(y|f)) \\; q(f) \\; df $$

        where \\(q(f) \\sim \\mathcal{N}(\\mu,\\Sigma)\\) and \\(p(y|f)\\) our likelihood. By default this uses Gauss-Hermite quadratures to approximate the integral.

        Args:
            y (torch.tensor): Values for y of shape (data_points,).
            mu (torch.tensor): Mean of the posterior \\(q(f)\\) of shape (data_points,1).
            var (torch.tensor): Variance of the posterior \\(q(f)\\) of shape (data_points,1).
            X (torch.tensor): Input of shape (data_points,input_dims) needed for multi-output likelihood.
        """
        # y,mu,var: Nx1
        q = self.quadrature(mu, var, lambda f: self.log_prob(y,f,X=X))  # Nx1
        return q.sum()  # sum over N

    def mean(self, f, X=None):
        """
        Calculate the mean of the likelihood.

        Args:
            f (torch.tensor): Posterior values for f of shape (data_points,input_dims).
            X (torch.tensor): Input of shape (data_points,input_dims) needed for multi-output likelihood.

        Returns:
            torch.tensor: Mean of the predictive posterior \\(p(y|f)\\) of shape (data_points,).
        """
        # f: NxM
        raise NotImplementedError()

    def variance(self, f, X=None):
        """
        Calculate the variance of the likelihood.

        Args:
            f (torch.tensor): Posterior values for f of shape (data_points,input_dims).
            X (torch.tensor): Input of shape (data_points,input_dims) needed for multi-output likelihood.

        Returns:
            torch.tensor: Variance of the predictive posterior \\(p(y|f)\\) of shape (data_points,).
        """
        # f: NxM
        raise NotImplementedError()

    def sample(self, f, X=None):
        """
        Sample from likelihood distribution.

        Args:
            f (torch.tensor): Samples for f of shape (data_points,n).
            X (torch.tensor): Input of shape (data_points,input_dims) needed for multi-output likelihood.

        Returns:
            torch.tensor: Samples of shape (data_points,n).
        """
        # f: nxN
        raise NotImplementedError()

    def predict(self, mu, var, ci=None, sigma=None, n=10000, X=None):
        """
        Calculate the mean and variance of the predictive distribution

        $$ \\mu = \\iint y \\; p(y|f) \\; q(f) \\; df dy $$
        $$ \\Sigma = \\iint y^2 \\; p(y|f) \\; q(f) \\; df dy - \\mu^2 $$

        where \\(q(f) \\sim \\mathcal{N}(\\mu,\\Sigma)\\) and \\(p(y|f)\\) our likelihood. By default this uses Gauss-Hermite quadratures to approximate both integrals.

        Args:
            mu (torch.tensor): Mean of the posterior \\(q(f)\\) of shape (data_points,1).
            var (torch.tensor): Variance of the posterior \\(q(f)\\) of shape (data_points,1).
            ci (list of float): Two percentages [lower, upper] in the range of [0,1] that represent the confidence interval.
            sigma (float): Number of standard deviations of the confidence interval. Only used for short-path for Gaussian likelihood.
            n (int): Number of samples used from distribution to estimate quantile.
            X (torch.tensor): Input of shape (data_points,input_dims) needed for multi-output likelihood.

        Returns:
            torch.tensor: Mean of the predictive posterior \\(p(y|f)\\) of shape (data_points,).
            torch.tensor: Lower confidence boundary of shape (data_points,).
            torch.tensor: Upper confidence boundary of shape (data_points,).
        """
        # mu,var: Nx1
        mu = self.quadrature(mu, var, lambda f: self.mean(f, X=X))
        if ci is None:
            return mu

        # TODO: naive way of estimating the quantiles, can we use MCMC or something more efficient?
        # TODO: is this correct???
        samples_f = torch.distributions.normal.Normal(mu, var).sample([n]) #nxN
        samples_y = self.sample(samples_f, X=X) # nxN
        #samples_y = self.sample(mu.T.repeat(n,1), X=X) # nxN
        samples_y, _ = samples_y.sort(dim=0)
        lower = int(ci[0]*n + 0.5)
        upper = int(ci[1]*n + 0.5)
        return mu, samples_y[lower,:], samples_y[upper,:]

class MultiOutputLikelihood(Likelihood):
    """
    Multi-output likelihood to assign a different likelihood per channel.

    Args:
        likelihoods (mogptk.gpr.likelihood.Likelihood): List of likelihoods equal to the number of output dimensions.
    """
    def __init__(self, *likelihoods):
        super().__init__()

        if isinstance(likelihoods, tuple):
            if len(likelihoods) == 1 and isinstance(likelihoods[0], list):
                likelihoods = likelihoods[0]
            else:
                likelihoods = list(likelihoods)
        elif not isinstance(likelihoods, list):
            likelihoods = [likelihoods]
        if len(likelihoods) == 0:
            raise ValueError("must pass at least one likelihood")
        for i, likelihood in enumerate(likelihoods):
            if not issubclass(type(likelihood), Likelihood):
                raise ValueError("must pass likelihoods")
            elif isinstance(likelihood, MultiOutputLikelihood):
                raise ValueError("can not nest MultiOutputLikelihoods")

        self.output_dims = len(likelihoods)
        self.likelihoods = torch.nn.ModuleList(likelihoods)

    def name(self):
        return '%s%s' % (self.__class__.__name__, [likelihood.name() for likelihood in self.likelihoods])

    def _channel_indices(self, X):
        c = X[:,0].long()
        m = [c==j for j in range(self.output_dims)]
        r = [torch.nonzero(m[j], as_tuple=False).reshape(-1) for j in range(self.output_dims)]
        return r

    def validate_y(self, y, X=None):
        if self.output_dims == 1:
            self.likelihoods[0].validate_y(y, X=X)
            return

        r = self._channel_indices(X)
        for i in range(self.output_dims):
            self.likelihoods[i].validate_y(y[r[i],:], X=X)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].log_prob(y[r[i],:], f[r[i],:])
        return res  # NxM

    def variational_expectation(self, y, mu, var, X=None):
        # y,mu,var: Nx1
        q = torch.tensor(0.0, dtype=config.dtype, device=config.device)
        r = self._channel_indices(X)
        for i in range(self.output_dims):
            q += self.likelihoods[i].variational_expectation(y[r[i],:], mu[r[i],:], var[r[i],:]).sum()  # sum over N
        return q

    def mean(self, f, X=None):
        # f: NxM
        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].mean(f[r[i],:])
        return res  # NxM

    def variance(self, f, X=None):
        # f: NxM
        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].variance(f[r[i],:])
        return res  # NxM

    def sample(self, f, X=None):
        # f: nxN
        r = self._channel_indices(X)
        for i in range(self.output_dims):
            f[:,r[i]] = self.likelihoods[i].sample(f[:,r[i]])
        return f  # nxN

    def predict(self, mu, var, ci=None, sigma=None, n=10000, X=None):
        # mu,var: Nx1
        r = self._channel_indices(X)
        res = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        if ci is None:
            for i in range(self.output_dims):
                res[r[i],:] = self.likelihoods[i].predict(mu[r[i],:], var[r[i],:], ci=ci, sigma=sigma, n=n)
            return res

        lower = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        upper = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:], lower[r[i],:], upper[r[i],:] = self.likelihoods[i].predict(mu[r[i],:], var[r[i],:], ci=ci, sigma=sigma, n=n)
        return res, lower, upper  # Nx1

    # TODO: predict is not possible?
    #def predict(self, mu, var, X=None, full=False):
    #    # mu: Nx1
    #    # var: Nx1 or NxN
    #    if self.output_dims == 1:
    #        return self.likelihoods[0].predict(mu,var,full=full)

    #    r = self._channel_indices(X)
    #    Ey = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
    #    Eyy = torch.empty(var.shape, device=config.device, dtype=config.dtype)
    #    if full:
    #        for i in range(self.output_dims):
    #            r1 = r[i].reshape(-1,1)
    #            r2 = r[i].reshape(1,-1)
    #            Ey[r[i],:], Eyy[r1,r2] = self.likelihoods[i].predict(mu[r[i],:], var[r1,r2], full=True)
    #    else:
    #        for i in range(self.output_dims):
    #            Ey[r[i],:], Eyy[r[i],:] = self.likelihoods[i].predict(mu[r[i],:], var[r[i],:], full=False)
    #    return Ey, Eyy-Ey.square()

class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood given by

    $$ p(y|f) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(y-f)^2}{2\\sigma^2}} $$

    with \\(\\sigma\\) the scale.

    Args:
        scale (float): Scale.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = Parameter(scale, lower=config.positive_minimum)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -0.5 * (np.log(2.0*np.pi) + 2.0*self.scale().log() + ((y-f)/self.scale()).square())
        return p  # NxM

    def variational_expectation(self, y, mu, var, X=None):
        # y,mu,var: Nx1
        p = -((y-mu).square() + var) / self.scale().square()
        p -= np.log(2.0 * np.pi)
        p -= 2.0*self.scale().log()
        return 0.5*p.sum()  # sum over N

    def mean(self, f, X=None):
        return f

    def variance(self, f, X=None):
        return self.scale().square()

    def sample(self, f, X=None):
        return torch.distributions.normal.Normal(f, scale=self.scale()).sample()

    def predict(self, mu, var, ci=None, sigma=None, n=10000, X=None):
        if ci is None and sigma is None:
            return mu

        if sigma is None:
            ci = torch.tensor(ci, device=config.device, dtype=config.dtype)
            lower = mu + torch.sqrt(2.0)*self.scale()*torch.erfinv(2.0*ci[0] - 1.0)
            upper = mu + torch.sqrt(2.0)*self.scale()*torch.erfinv(2.0*ci[1] - 1.0)
        else:
            lower = mu - sigma*self.scale()
            upper = mu + sigma*self.scale()
        return mu, lower, upper

class StudentTLikelihood(Likelihood):
    """
    Student's t likelihood given by

    $$ p(y|f) = \\frac{\\Gamma\\left(\\frac{\\nu+1}{2}\\right)}{\\Gamma\\left(\\frac{\\nu}{2}\\right)\\sqrt{\\pi\\nu}\\sigma} \\left( 1 + \\frac{(y-f)^2}{\\nu\\sigma^2} \\right)^{-(\\nu+1)/2} $$

    with \\(\\nu\\) the degrees of freedom and \\(\\sigma\\) the scale.

    Args:
        dof (float): Degrees of freedom.
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, dof=3, scale=1.0, quadratures=20):
        super().__init__(quadratures)

        self.dof = torch.tensor(dof, device=config.device, dtype=config.dtype)
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -0.5 * (self.dof+1.0)*torch.log1p(((y-f)/self.scale()).square()/self.dof)
        p += torch.lgamma((self.dof+1.0)/2.0)
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * torch.log(self.dof*np.pi*self.scale().square())
        return p  # NxM

    def mean(self, f, X=None):
        if self.dof <= 1.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f

    def variance(self, f, X=None):
        if self.dof <= 2.0:
            return np.nan
        return self.scale().square() * self.dof/(self.dof-2.0)

    def sample(self, f, X=None):
        return torch.distributions.studentT.StudentT(self.dof, f, self.scale()).sample()

class ExponentialLikelihood(Likelihood):
    """
    Exponential likelihood given by

    $$ p(y|f) = h(f) e^{-h(f)y} $$

    with \\(h\\) the link function and \\(y \\in [0.0,\\infty)\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)

        self.link = link

    def validate_y(self, y, X=None):
        if torch.any(y < 0.0):
            raise ValueError("y must be positive")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        if self.link == exp:
            p = -y/self.link(f) - f
        else:
            p = -y/self.link(f) - self.link(f).log()
        return p  # NxM

    def mean(self, f, X=None):
        return self.link(f)

    def variance(self, f, X=None):
        return self.link(f).square()

    def sample(self, f, X=None):
        rate = self.link(f)
        return torch.distributions.exponential.Exponential(rate).sample()

class LaplaceLikelihood(Likelihood):
    """
    Laplace likelihood given by

    $$ p(y|f) = \\frac{1}{2\\sigma}e^{-\\frac{|y-f|}{\\sigma}} $$

    with \\(\\sigma\\) the scale.

    Args:
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, quadratures=20):
        super().__init__(quadratures)

        self.scale = Parameter(scale, lower=config.positive_minimum)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxM

    def mean(self, f, X=None):
        return f

    def variance(self, f, X=None):
        return 2.0 * self.scale().square()

    def sample(self, f, X=None):
        return torch.distributions.laplace.Laplace(f, self.scale()).sample()

class BernoulliLikelihood(Likelihood):
    """
    Bernoulli likelihood given by

    $$ p(y|f) = h(f)^k (1-h(f))^{n-k} $$

    with \\(h\\) the link function, \\(k\\) the number of \\(y\\) values equal to 1, \\(n\\) the number of data points, and \\(y \\in \\{0.0,1.0\\}\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=inv_probit, quadratures=20):
        super().__init__(quadratures)

        self.link = link

    def validate_y(self, y, X=None):
        if torch.any((y != 0.0) & (y != 1.0)):
            raise ValueError("y must have only 0.0 and 1.0 values")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = self.link(f)
        return torch.log(torch.where(0.5 <= y, p, 1.0-p))  # NxM

    def mean(self, f, X=None):
        return self.link(f)

    def variance(self, f, X=None):
        return self.link(f) - self.link(f).square()

    def sample(self, f, X=None):
        return f

    def predict(self, mu, var, ci=None, sigma=None, n=10000, X=None):
        if self.link != inv_probit:
            return super().predict(mu, var, ci=ci, sigma=sigma, n=n, X=X)

        p = self.link(mu / torch.sqrt(1.0 + var))
        if ci is None and sigma is None:
            return p
        return p, p, p

class BetaLikelihood(Likelihood):
    """
    Beta likelihood given by

    $$ p(y|f) = \\frac{\\Gamma(\\sigma)}{\\Gamma\\left(h(f)\\sigma\\right)\\Gamma\\left((1-h(f)\\sigma\\right)} y^{h(f)\\sigma} (1-y)^{(1-h(f))\\sigma} $$

    with \\(h\\) the link function, \\(\\sigma\\) the scale, and \\(y \\in (0.0,1.0)\\).

    Args:
        scale (float): Scale.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, link=inv_probit, quadratures=20):
        super().__init__(quadratures)

        self.link = link
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def validate_y(self, y, X=None):
        if torch.any((y <= 0.0) | (1.0 <= y)):
            raise ValueError("y must be in the range (0.0,1.0)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        mixture = self.link(f)
        alpha = mixture * self.scale()
        beta = (1.0-mixture) * self.scale()

        p = (alpha-1.0)*y.log()
        p += (beta-1.0)*torch.log1p(-y)
        p += torch.lgamma(alpha+beta)
        p -= torch.lgamma(alpha)
        p -= torch.lgamma(beta)
        return p  # NxM

    def mean(self, f, X=None):
        return self.link(f)

    def variance(self, f, X=None):
        mixture = self.link(f)
        return (mixture - mixture.square()) / (self.scale() + 1.0)

    def sample(self, f, X=None):
        mixture = self.link(f)
        alpha = mixture * self.scale()
        beta = (1.0-mixture) * self.scale()
        return torch.distributions.beta.Beta(alpha, beta).sample()

class GammaLikelihood(Likelihood):
    """
    Gamma likelihood given by

    $$ p(y|f) = \\frac{1}{\\Gamma(k)h(f)^k} y^{k-1} e^{-y/h(f)} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\). 

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)

        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, y, X=None):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -y/self.link(f)
        p += (self.shape()-1.0)*y.log()
        p -= torch.lgamma(self.shape())
        if self.link == exp:
            p -= self.shape()*f
        else:
            p -= self.shape()*self.link(f).log()
        return p  # NxM

    def mean(self, f, X=None):
        return self.shape()*self.link(f)

    def variance(self, f, X=None):
        return self.shape()*self.link(f).square()

    def sample(self, f, X=None):
        rate = 1.0/self.link(f)
        return torch.distributions.gamma.Gamma(self.shape(), rate).sample()

class PoissonLikelihood(Likelihood):
    """
    Poisson likelihood given by

    $$ p(y|f) = \\frac{1}{y!} h(f)^y e^{-h(f)} $$

    with \\(h\\) the link function and \\(y \\in \\mathbb{N}_0\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)
        # TODO: the variational_expectation is tractable?

        self.link = link

    def validate_y(self, y, X=None):
        if torch.any(y < 0.0):
            raise ValueError("y must be in the range [0.0,inf)")
        if not torch.all(y == y.long()):
            raise ValueError("y must have integer count values")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        if self.link == exp:
            p = y*f
        else:
            p = y*self.link(f).log()
        p -= torch.lgamma(y+1.0)
        p -= self.link(f)
        return p  # NxM

    def mean(self, f, X=None):
        return self.link(f)

    def variance(self, f, X=None):
        return self.link(f)

    def sample(self, f, X=None):
        rate = self.link(f)
        return torch.distributions.poisson.Poisson(rate).sample()

class WeibullLikelihood(Likelihood):
    """
    Weibull likelihood given by

    $$ p(y|f) = \\frac{k}{h(f)} \\left( \\frac{y}{h(f)} \\right)^{k-1} e^{-(y/h(f))^k} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\).

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)

        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, y, X=None):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        if self.link == exp:
            p = -self.shape()*f
        else:
            p = -self.shape()*self.link(f).log()
        p += self.shape().log() + (self.shape()-1.0)*y.log()
        p -= (y/self.link(f))**self.shape()
        return p  # NxM

    def mean(self, f, X=None):
        return f * torch.lgamma(1.0 + 1.0/self.shape()).exp()

    def variance(self, f, X=None):
        a = torch.lgamma(1.0 + 2.0/self.shape()).exp()
        b = torch.lgamma(1.0 + 1.0/self.shape()).exp()
        return f.square() * (a - b.square())

    def sample(self, f, X=None):
        scale = self.link(f)
        return torch.distributions.weibull.Weibull(scale, self.shape()).sample()

class LogLogisticLikelihood(Likelihood):
    """
    Log-logistic likelihood given by

    $$ p(y|f) = \\frac{(k/h(f)) (y/h(f))^{k-1}}{\\left(1 + (y/h(f))^k\\right)^2} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\).

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)

        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, y, X=None):
        if torch.any(y < 0.0):
            raise ValueError("y must be in the range [0.0,inf)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        if self.link == exp:
            p = -self.shape()*f
        else:
            p = -self.shape()*self.link(f).log()
        p -= 2.0*torch.log1p((y/self.link(f))**self.shape())
        p += self.shape().log()
        p += (self.shape()-1.0)*y.log()
        return p  # NxM

    def mean(self, f, X=None):
        return self.link(f) / torch.sinc(1.0/self.shape())

    def variance(self, f, X=None):
        if self.shape() <= 2.0:
            return np.nan
        a = 1.0/torch.sinc(2.0/self.shape())
        b = 1.0/torch.sinc(1.0/self.shape())
        return self.link(f).square() * (a - b.square())

    def sample(self, f, X=None):
        raise NotImplementedError()

class LogGaussianLikelihood(Likelihood):
    """
    Log-Gaussian likelihood given by

    $$ p(y|f) = \\frac{1}{y\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(\\log(y) - f)^2}{2\\sigma^2}} $$

    with \\(\\sigma\\) the scale and \\(y \\in (0.0,\\infty)\\).

    Args:
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, quadratures=20):
        super().__init__(quadratures)

        self.scale = Parameter(scale, lower=config.positive_minimum)

    def validate_y(self, y, X=None):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        logy = y.log()
        p = -0.5 * (np.log(2.0*np.pi) + 2.0*self.scale().log() + ((logy-f)/self.scale()).square())
        p -= logy
        return p  # NxM

    def mean(self, f, X=None):
        return torch.exp(f + 0.5*self.scale().square())

    def variance(self, f, X=None):
        return (self.scale().square().exp() - 1.0) * torch.exp(2.0*f + self.scale().square())

    def sample(self, f, X=None):
        return torch.distributions.log_normal.LogNormal(f, self.scale()).sample()

class ChiSquaredLikelihood(Likelihood):
    """
    Chi-squared likelihood given by

    $$ p(y|f) = \\frac{1}{2^{f/2}\\Gamma(f/2)} y^{f/2-1} e^{-y/2} $$

    with \\(y \\in (0.0,\\infty)\\).

    Args:
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, quadratures=20):
        super().__init__(quadratures)

    def validate_y(self, y, X=None):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -0.5*f*np.log(2.0) - torch.lgamma(f/2.0) + (f/2.0-1.0)*y.log() - 0.5*y
        return p  # NxM

    def mean(self, f, X=None):
        return f

    def variance(self, f, X=None):
        return 2.0*f

    def sample(self, f, X=None):
        raise NotImplementedError()
