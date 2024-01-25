import numpy as np
import scipy.integrate as spi
from tqdm import tqdm
from numba import njit

G = 4.3009125E-6 # kpc (km/s)^2 /Msun
kms_to_kpcMyr = 1.023e-3 # kpc/Myr
np.random.seed(1234)

#Mass Dependent Mass Loss Rate
#@njit
def Mdot(m0,td,mdep,tobs,tau):
	return (-m0/((1-mdep)*td)) * (1-((tobs-(1/tau))/td))**(mdep/(1-mdep))

# Cluster Mass at Time t
#@njit
def Mt(m0,td,mdep,tobs,tau):
	return m0*((1-((tobs-(1/tau))/td)))**(1/(1-mdep))

# QSG Density integrand
#@njit
def drho_dtau(tau,tobs,td,phi1,m0,mbh,mdep,orbit_params,free_params):
	R,Vc,Vc_kpc_Myr,AngVel,gamma = orbit_params
	fe,eps = free_params
	mt = Mt(m0-mbh, td, mdep, tobs, tau) + mbh
	#mt = Mt(m0, td, mdep, tobs, tau) + mbh
	#if (tobs - 1/tau) <= td and mt >= mbh:
	if (tobs - 1/tau) <= td and mt >= mbh:# and tau <= 0.1:
		dm = Mdot(m0, td, mdep, tobs, tau)
		rj = (G*mt/(2*(Vc/R)**2))**(1/3)
		rh = 0.15 * rj # assume a roche filling cluster
		aP = 1.305 * rh
		dr0 = fe * rj
		sig = (G*mt/(6*aP))**(1/2) * (1+(fe/(0.15*1.305))**2)**(-1/4) * 1.023e-3 # kpc/Myr
		A = (gamma**2/(4-gamma**2)) * abs(dm)*R/np.sqrt(2*np.pi*sig**2)
		B = (gamma**2/(4-gamma**2)) * R/np.sqrt(2*sig**2)
		C = ((1+eps)/np.sqrt(2*sig**2)) * dr0/R * Vc_kpc_Myr
		return (A/tau) * np.exp(-(B*phi1*tau + C)**2)
	else:
		return 0

#@njit
def QSG_AN(t,td,m0,mdep,R,Vc,mbh=0,gamma=np.sqrt(2),fe=1.5,eps=0.58):
	'''
	Quantifying Stream Growth - Analytic
	Parameters:
		t     -  Observation time (time since the start of stripping) [Myr]
		td    -  Dissolution time [Myr]
		m0    -  Initial Cluster Mass [Msun]
		mdep  -  Mass dependency of the mass-loss rate (eta in Roberts et al. 2024)
		R     -  Galactocentric radius [kpc]
		Vc    -  Circular velocity [km/s]
		mbh   -  Black hole mass at dissolution (default : 0) [Msun]
		gamma -  epicycic frequency / angular frequency
		fe    -  Free parameter : escape radius / jacobi radius
		eps   -  Free parameter relating mean escape velocity to escape radius
	Output:
		phi1_arr - array of phi_1 coordinates [deg]
		rho_arr  - array of density values [Msun/deg]
	'''
	orbit_params = np.array([R,Vc,Vc*kms_to_kpcMyr,R/Vc/kms_to_kpcMyr,gamma])
	free_params = np.array([fe,eps])
	phi1_arr = -np.linspace(0,np.pi,201,endpoint=True)
	rho_arr = np.empty(np.shape(phi1_arr))
	tlo = 1/t
	if t <= td:
		#tup = np.inf # Work out why the integral fails
		tup = 1/10
	else:
		tup = 1/(t-td)
	for i in range(len(phi1_arr)):
		rho_arr[i] = spi.quad(drho_dtau,tlo,tup,args=(t,td,phi1_arr[i],m0,mbh,mdep,orbit_params,free_params),limit=200)[0]# Msun/rad #/(R*1e3) # Msun/pc
	phi1_arr = np.concatenate(((phi1_arr[1:])[::-1],-phi1_arr)) *180/np.pi # [-180 , 180]
	rho_arr = np.concatenate(((rho_arr[1:])[::-1],rho_arr))/2 * np.pi/180 # Msun/deg
	return phi1_arr, rho_arr

# QSG Equations of motion
#@njit
def delta_phi1(t,dvx,dvy,dr,orbit_params,free_params):
	R,Vc,Vc_kpc_Myr,AngVel,gamma = orbit_params
	fe,eps = free_params
	LinearTerms = - (4-gamma**2)/gamma**2 * np.multiply((dvy + (1+eps) * (dr/R) * Vc_kpc_Myr), t/R)
	OscillatoryTerm1 = - 2/gamma**3 * (gamma**2 - 2 - 2*eps) * np.multiply(dr/R, np.sin((gamma*Vc_kpc_Myr*t/R)))
	OscillatoryTerm2 = 4/gamma**3 * np.multiply(dvy/Vc_kpc_Myr, np.sin((gamma*Vc_kpc_Myr*t/R)))
	OscillatoryTerm3 = - 2/gamma**2 * np.multiply(dvx/Vc_kpc_Myr, (1-np.cos((gamma*Vc_kpc_Myr*t/R))))
	return LinearTerms + OscillatoryTerm1 + OscillatoryTerm2 + OscillatoryTerm3
#@njit
def delta_r(t,dvx,dvy,dr,orbit_params,free_params):
	R,Vc,Vc_kpc_Myr,AngVel,gamma = orbit_params
	fe,eps = free_params
	Term1 = np.multiply(dr, np.cos((gamma*Vc_kpc_Myr*t/R)))
	Term2 = (2*R/gamma**2) * np.multiply(((dvy/Vc_kpc_Myr) + (1+eps)*(dr/R)), (1-np.cos((gamma*Vc_kpc_Myr*t/R))))
	Term3 = R*np.multiply(dvx/Vc_kpc_Myr, (np.sin((gamma*Vc_kpc_Myr*t/R)))) / gamma
	return Term1+Term2+Term3
#@njit
def delta_z(t,dvz,orbit_params,free_params):
	AngVel = orbit_params[3]
	return dvz/AngVel * np.sin(AngVel*t)

def PS_genICs(m0,td,mdep,mstar,mbh,orbit_params,free_params):
	R,Vc,Vc_kpc_Myr,AngVel,gamma = orbit_params
	fe,eps = free_params
	Nstars = int(m0/mstar)
	Ms = np.ones(Nstars)*mstar # Star Masses # single mass to not have to deal with low mass stars preferentially escaping
	Mts = np.cumsum(Ms[::-1])[::-1] # Cluster Mass at escape times
	tesc = td * m0**(mdep-1)*(m0**(1-mdep) - (Mts)**(1-mdep)) # escape times
	rjs = (G*(Mts+mbh)/(2*(Vc/R)**2))**(1/3) # Escape radii
	rhs = 0.15 * rjs # Half-mass radii # 0.15 from Henon 1961
	aPs = 1.305*rhs
	xes = fe * rjs # Escape radius
	sigs = (G*Mts/(6*aPs))**(1/2) * (1+(fe/(0.15*1.305))**2)**(-1/4) * 1.023e-3 # kpc/Myr # vel dis at xes Heggie and Hut 2003
	dvxs = np.random.normal(0,1,size=Nstars) * sigs
	dvys = np.random.normal(0,1,size=Nstars) * sigs
	dvzs = np.random.normal(0,1,size=Nstars) * sigs
	xes[1::2] *= -1
	dvxs[1::2] *= -1
	dvys[1::2] *= -1
	dvzs[1::2] *= -1
	starsarr = np.zeros((Nstars,8)) # M x0, y0, z0, dvx, dvy, dvz, tesc
	starsarr[:,0] = Ms
	starsarr[:,1] = xes
	starsarr[:,4] = dvxs
	starsarr[:,5] = dvys
	starsarr[:,6] = dvzs
	starsarr[:,7] = tesc
	return starsarr


def QSG_PS(trange,td,m0,mstar,mdep,R,Vc,mbh=0,gamma=np.sqrt(2),fe=1.5,eps=0.58):
	'''
	Quantifying Stream Growth - Particle Spray
	Parameters:
		trange -  Params for creating observation time array [start,stop,step]
		       -  If single value then this will be taken as the observation time
		       -  Observation time (time since the start of stripping) [Myr]
		td     -  Dissolution time [Myr]
		m0     -  Initial Cluster Mass [Msun]
		mdep   -  Mass dependency of the mass-loss rate (eta in Roberts et al. 2024)
		R      -  Galactocentric radius [kpc]
		Vc     -  Circular velocity [km/s]
		mbh    -  Black hole mass at dissolution (default : 0) [Msun]
		gamma  -  epicycic frequency / angular frequency
		fe     -  Free parameter : escape radius / jacobi radius
		eps    -  Free parameter relating mean escape velocity to escape radius
	Output:
		snaps   - observation times
		StarPos - array of star positions [kpc]
		        - (nsnaps,nstars,[x,y,z,Mstar]) [kpc,kpc,kpc,Msun]
	'''
	orbit_params = np.array([R,Vc,Vc*kms_to_kpcMyr,R/Vc/kms_to_kpcMyr,gamma])
	free_params = np.array([fe,eps])
	stars = PS_genICs(m0,td,mdep,mstar,mbh,orbit_params,free_params)
	if len(np.atleast_1d(trange)) == 1:
		snaps = np.atleast_1d(trange)
	elif len(np.atleast_1d(trange)) > 1 and len(np.atleast_1d(trange)) != 3:
		snaps = trange
	else:
		snaps = np.arange(trange[0],trange[1]+trange[2],trange[2])
	StarPos = []
	StarPos = np.empty((len(snaps),len(stars),4))
	StarPos.fill(np.NaN)
	for i in range(len(snaps)):
		released = stars[stars[:,-1] <= snaps[i]]
		t_ev = - released[:,-1] + snaps[i]
		x0 = released[:,1]
		y0 = released[:,2]
		dvx = released[:,3]
		dvy = released[:,4]
		dvz = released[:,5]
		xt = delta_r(t_ev,dvx,dvy,x0,orbit_params,free_params)
		yt = delta_phi1(t_ev,dvx,dvy,x0,orbit_params,free_params)*R
		zt = delta_z(t_ev,dvz,orbit_params,free_params)
		StarPos[i,:len(released)] = np.array([xt,yt,zt,released[:,0]]).T
	if len(np.atleast_1d(trange)) == 1:
		StarPos = StarPos[0]
	return snaps,StarPos
