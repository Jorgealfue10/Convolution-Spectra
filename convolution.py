import numpy as np
from astropy.modeling import models
import astropy.units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import fit_lines
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
import sys


nfile = int(sys.argv[1])
amp = float(sys.argv[-2])
png_name = sys.argv[-1]

if len(sys.argv[:]) > 4+nfile:
    globalmin = float(sys.argv[-4])
    globalmax = float(sys.argv[-3])



def lorentzian(x, a, x0):
    return 1/(np.pi)*((1/2*a)/((x-x0)**2+(1/2*a)**2))

std=1.5

fig1,ax1 = plt.subplots(figsize=(10,10))

for m in range(nfile):
    data = np.loadtxt("./"+sys.argv[m+2])

    emin = np.min(data[:,2])
    emax = np.max(data[:,2])

    npoints = int((emax+500)-(emin-500))
    x = np.zeros((data.shape[0], npoints))
    y = np.zeros((data.shape[0], npoints))

    falsedat = data[:,2]-emin+100

    for i in range(falsedat.shape[0]):
        x0 = falsedat[i]
        maxintens = lorentzian(x0,amp,x0)
        for j in range(0,npoints):
            diff = abs(j-x0)
            if diff < 0.1:
                x[i,j] = x0
            else:
                x[i,j] = j
            y[i,j] = lorentzian(x[i,j], amp, x0)/maxintens*data[i,4]

    # if globalmin is None:
    #     globalmin = emin
    # else:
    #     if globalmin > emin:
    #         globalmin = emin

    # if globalmax is None:
    #     globalmax = emax
    # else:
    #     if globalmax < emax:
    #         globalmax = emax

    x = x+emin-100

    funct = np.zeros((npoints))
    for i in range(npoints):
        funct[i] = np.max(y[:,i])

    kernel = Gaussian1DKernel(stddev=std)
    convoluted = convolve(funct,kernel,normalize_kernel=True,boundary='extend')
    convoluted = convoluted/np.max(convoluted)*np.max(data[:,4])

    ax1.plot(x[1,:],convoluted,alpha=1)
    ax1.fill_between(x[1,:],0,convoluted,alpha=0.2)

    if len(sys.argv[:]) > 4+nfile:
        ax1.set_xlim(globalmin,globalmax)

ax1.tick_params(axis='both', which='major', labelsize=30)
ax1.set_xlabel(r" $\nu$ ($cm^{-1}$)",fontsize=30)
ax1.set_ylabel(r" I (a.u.)",fontsize=30)

fig1.savefig(png_name+".png",dpi=300,transparent=True)
