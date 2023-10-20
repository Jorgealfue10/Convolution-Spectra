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
rots = sys.argv[-4]
labcond = sys.argv[-3]
amp = float(sys.argv[-2])
png_name = sys.argv[-1]

r=6

if len(sys.argv[:]) > r+nfile:
    globalmin = float(sys.argv[-6])
    globalmax = float(sys.argv[-5])



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

    x = x+emin-100

    funct = np.zeros((npoints))
    for i in range(npoints):
        funct[i] = np.max(y[:,i])

    kernel = Gaussian1DKernel(stddev=std)
    convoluted = convolve(funct,kernel,normalize_kernel=True,boundary='extend')
    convoluted = convoluted/np.max(convoluted)*np.max(data[:,4])

    ax1.plot(x[1,:],convoluted,alpha=1)
    
    if labcond == "Temp":
        lab="T = "
        for i in sys.argv[m+2]:
            if i == "/":
                break
            elif i == "T":
                continue
            else:
                lab = lab + i
        ax1.fill_between(x[1,:],0,convoluted,alpha=0.2,label=lab+" K")
    else:
        lab=""
        check = False
        for i in sys.argv[m+2]:
            if i == "/":
                check = True
                continue
            if check:
                if i == ".":
                    check = False
                    continue
                if i != "_":
                    lab = lab+i
                else:
                    lab = lab + "->"
            else:
                continue
        ax1.fill_between(x[1,:],0,convoluted,alpha=0.2,label=lab)
        
    if len(sys.argv[:]) > r+nfile:
        ax1.set_xlim(globalmin,globalmax)

    if rots == "yes":
        ax1.vlines(data[:,2],0,data[:,4])
        for i in range(len(data[:,0])):
            if data[i,4] > 0.1e-20: #np.max(data[i,4])/100:
                if data[i,0] != data[i,1]:
                   ax1.annotate(str(data[i,1])+r"->"+str(data[i,0]),xy=(data[i,2]-8,data[i,4]+0.5e-21),rotation=90,fontsize=15)
                if data[i,0] == data[i,1] and data[i,1] == 0.5:
                   ax1.annotate(str(data[i,1])+r"->"+str(data[i,0]),xy=(data[i,2]-8,data[i,4]+0.5e-21),rotation=90,fontsize=15)

ax1.tick_params(axis='both', which='major', labelsize=30)
ax1.set_xlabel(r" $\nu$ ($cm^{-1}$)",fontsize=30)
ax1.set_ylabel(r" I (cm/mol)",fontsize=30)

ax1.legend(fontsize=30,frameon=False)

fig1.tight_layout()

fig1.savefig(png_name+".png",dpi=300,transparent=True)
