import numpy as np
import matplotlib.pyplot as plt
from mogptk import Data, Model
from mogptk.gpr.likelihood import LaplaceLikelihood, GaussianLikelihood, \
    StudentTLikelihood
from mogptk.gpr import ConstantMean
from mogptk import Hensman, OpperArchambeau
from mogptk.gpr.singleoutput import SquaredExponentialKernel, PeriodicKernel, \
    WhiteKernel

np.random.seed(1410)

#=======================#
#========= Data ========#
#=======================#
# input 1D
nt = 10**3
a, b = -20., 20.
t_regular = np.linspace(-10., 10., num=nt)
t_irreg = np.random.uniform(a, b, size=nt).sort()
y_useless = np.random.normal(size=nt)
f0 = 5.0

# just to instantiate a mogptk.Model, independent of input
data = Data(X=t_regular, Y=y_useless, y_label='placeholder')

#=======================#
#======== Model ========#
#=======================#
ker = PeriodicKernel(input_dims=1)
ker.period.assign(value=[f0], train=False)
ker.lengthscale.assign(value=[1.0], train=False)

ker = WhiteKernel()

# for any kernel
ker.magnitude.assign(value=[1.0], train=False)

scale = 0.1
lik_hvtail = StudentTLikelihood(scale=scale, dof=0.5, quadratures=100)
#LaplaceLikelihood(scale=scale, quadratures=100) 
lik_gaus = GaussianLikelihood(scale=scale)

model_hvtail = Model(
    dataset=data,
    kernel=ker,
    mean=ConstantMean(),
    inference=Hensman(likelihood=lik_hvtail)
)
model_gaus = Model(
    dataset=data,
    kernel=ker,
    mean=ConstantMean(),
    inference=Hensman(likelihood=lik_gaus)
)

#=======================#
#====== Sampling =======#
#=======================#
# number of processes
n_proc = 5
obs_hvtail = model_hvtail.gpr.sample_y(
    Z=t_regular.reshape((-1, 1)),
    n=n_proc
)

obs_gaus = model_gaus.gpr.sample_y(
    Z=t_regular.reshape((-1, 1)),
    n=n_proc
)
#=======================#
#======== Plot =========#
#=======================#
fig, ax = plt.subplots(n_proc, 2, sharex=True, sharey=True)
marker_size = 0.75

for i in range(n_proc):

    ax[i, 0].plot(
        t_regular,
        obs_hvtail[i].numpy().T,
        linestyle='None',
        marker='.',
        markersize=marker_size
    )
    ax[i, 1].plot(
        t_regular,
        obs_gaus[i].numpy().T,
        linestyle='None',
        marker='.',
        markersize=marker_size
    )

    ax[i, 0].set_ylim((-8., 8.))
    ax[i, 1].set_ylim((-8., 8.))

ax[0, 0].set_title(lik_hvtail.__class__())
ax[0, 1].set_title(lik_gaus.__class__())

# ax.set_xlabel('time')


fig.savefig('./gp_laplace_lik.png')