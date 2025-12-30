from mogptk.gpr.likelihood import BernoulliLikelihood, GammaLikelihood
from mogptk.gpr.singleoutput import SquaredExponentialKernel
from mogptk import Laplace
from mogptk.gpr.mean import ConstantMean
from mogptk import Model
from mogptk import DataSet, Data
import torch

torch.manual_seed(2025)

#-- Data generation --#
n = 500
x = torch.distributions.Uniform(0., 5.).sample((n, 1))
y = torch.tensor(x > 2.5, dtype=torch.int)
mogptk_data = DataSet([x.numpy()], [y.numpy()])
mogptk_data[0].remove_randomly(pct=0.30)

#-- Model with non-Gaussian likelihood and Laplace inference method --#
ker = SquaredExponentialKernel()
model_nongauss = Model(
    dataset=mogptk_data,
    kernel=ker,
    mean=ConstantMean(),
    inference=Laplace(likelihood=BernoulliLikelihood())
)

#-- Training --#
fig, ax = model_nongauss.train(method='adam', iters=10**3, verbose=True, plot=True)
fig.savefig('./bern_laplace_inf.png')