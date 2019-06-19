from math import floor
from numpy import arange, exp, pi, flipud, append, conj, real
from numpy.fft import ifft
from numpy.random import randn, rand

def powernoise(alpha, N, randpower=False, normalize=False):
	"""
	Generate samples of power law noise. The power spectrum
	of the signal scales as f^(-alpha).

	Usage:
		x = powernoise(alpha, N)
		x = powernoise(alpha, N, randpower=True, normalize=True)

	Inputs:
		alpha - power law scaling exponent
		N     - number of samples to generate

	Output:
		x     - N x 1 vector of power law samples

	By default, the power spectrum is deterministic, and the phases are
	uniformly distributed in the range -pi to +pi. The power law extends
	all the way down to 0Hz (DC) component. By specifying randpower=True
	the power spectrum will be stochastic with Chi-square distribution.
	With normalize=True the output is scaled to the range [-1, 1], and
	consequently the power law will not necessarily extend right down to 0Hz.

	Original Matlab code by Max Little:
	Little MA et al. (2007), "Exploiting nonlinear recurrence and fractal
	scaling properties for voice disorder detection", Biomed Eng Online, 6:23
	See http://www.maxlittle.net/software/
	"""

	N2 = floor(N/2)-1
	f = arange(2, N2+2)
	A2 = 1./(f**(alpha/2))

	if (randpower):
		p2 = (rand(N2)-0.5)*2.*pi
		d2 = A2 * exp(1j*p2)
	else:
		# 20080323
		p2 = randn(N2) + 1j * randn(N2)
		d2 = A2 * p2


	d = append( append( append([1.], d2), [1/((N2+2)**alpha)] ), flipud(conj(d2)) )
	x = real(ifft(d))

	if (normalize):
		x = ((x - min(x))/(max(x) - min(x)) - 0.5) * 2

	return x