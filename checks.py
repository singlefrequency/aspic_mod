import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy
from scipy.misc import derivative
plt.rcParams.update({'text.usetex':True})

class Potentials:
    def V_RCHI(empty,A1,M,phi):
        Mp = 1
        return M**4*(1-2*np.exp(-2*phi/(np.sqrt(6)/Mp))+A1/(16*np.pi**2)*phi/(np.sqrt(6)*Mp))

    def V_LFI(empty,p,M,phi):
        Mp = 1
        return M**4*(phi/Mp)**p

    def V_GMLFI(empty,alpha,M,phi):
        Mp = 1
        return M**4*(phi/Mp)**2*(1+alpha*phi**2/Mp**2)

    def V_RCMI(empty,alpha,M,phi):
        Mp = 1
        return M**4*(phi/Mp)**2*(1-2*alpha*phi**2/Mp**2*np.log(phi/Mp))

    def V_RCQI(empty,alpha,M,phi):
        Mp = 1
        return M**4*(phi/Mp)**4*(1-alpha*np.log(phi/Mp))

    def V_NI(empty,f,M,phi):
        return M**4*(1+np.cos(phi/f))
        
    def V_ESI(empty,q,M,phi):
        Mp = 1
        return M**4*(1-np.exp(-q*phi/Mp))

    def V_KMII(empty,alpha,M,phi):
        Mp = 1
        return M**4*(1-alpha*phi/Mp*np.exp(-phi/Mp))

    def V_HF1I(empty,A1,M,phi):
        Mp = 1
        return M**4*(1+A1*phi/Mp)**2*(1-2/3*(A1/(1+A1*phi/Mp))**2)

    def V_CWI(empty,Q,M,phi):
        Mp = 1
        alpha = 4*np.exp(1)
        return M**4*(1+alpha*(phi/Q)**4*np.log(phi/Q))

    def V_LI(empty,alpha,M,phi):
        Mp = 1
        return M**4*(1+alpha*np.log(phi/Mp))

    def V_RPI(empty,p,M,phi):
        Mp = 1
        return M**4*np.exp(-2*np.sqrt(2/3)*phi/Mp)*np.abs(np.exp(np.sqrt(2/3)*phi/Mp)-1)**(2*p/(2*p-1))

    def V_DWI(empty,phi0,M,phi):
        Mp = 1
        return M**4*((phi/phi0)**2-1)**2

    def V_MHI(empty,mu,M,phi):
        Mp = 1
        return M**4*(1-1/np.cosh(phi/mu))

    def V_RGI(empty,alpha,M,phi):
        Mp = 1
        return M**4*((phi/Mp)**2)/(alpha+(phi/Mp)**2)

    def V_GMSSMI(empty,phi0,M,phi):
        return M**4*((phi/phi0)**2-2/3*(phi/phi0)**6+1/5*(phi/phi0)**10)

    def V_GRIPI(empty,phi0,M,phi):
        return M**4*((phi/phi0)**2-4/3*(phi/phi0)**3+1/2*(phi/phi0)**4)

    def V_AI(empty,mu,M,phi):
        return M**4*(1-2/np.pi*np.arctan(phi/mu))

    def V_CNAI(empty,alpha,M,phi):
        Mp = 1
        return M**4*(3-(3+alpha**2)*np.tanh(alpha/np.sqrt(2)*phi/Mp)**2)

    def V_CNBI(empty,alpha,M,phi):
        Mp = 1
        return M**4*((3-alpha**2)*np.tan(alpha/np.sqrt(2)*phi/Mp)**2-3)

    def V_OSTI(empty,phi0,M,phi):
        return -M**4*(phi/phi0)**2*np.log((phi/phi0)**2)

    def V_WRI(empty,phi0,M,phi):
        return M**4*np.log(phi/phi0)**2

    def V_SFI1(empty,mu,M,phi):
        p = 1
        return M**4*(1-(phi/mu)**p)

    def V_SFI2(empty,mu,M,phi):
        p = 2
        return M**4*(1-(phi/mu)**p)

    def V_SFI4(empty,mu,M,phi):
        p = 4
        return M**4*(1-(phi/mu)**p)
    
def eps1(par,M,phi,V):
    Mp = 1
    return Mp**2/2*(np.gradient(V(par,M,phi),phi)/V(par,M,phi))**2

def extract_data(V):
    model = V.lower()    
    if model == 'gmlfi':
        newname = 'mlfi'
        par =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,0]
        phiend =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,1]
        M =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,2]
        dtphi =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,3]
    elif model == 'rpi':
        newname = 'rpi3'
        par =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,0]
        phiend =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,1]
        M =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,2]
        dtphi =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,3]
    elif model == 'gmssmi':
        newname = 'mssmi'
        par =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,0]
        phiend =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,1]
        M =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,2]
        dtphi =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,3]
    elif model == 'gripi':
        newname = 'ripi'
        par =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,0]
        phiend =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,1]
        M =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,2]
        dtphi =  np.loadtxt('aspic/src/' + model + '/' + newname + '_true.dat')[:,3]
    elif model == 'sfi1':
        newname1 = 'sfi'
        newname2 = 'sfip1'
        par =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,1]
        phiend =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,2]
        M =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,3]
        dtphi =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,4]
    elif model == 'sfi2':
        newname1 = 'sfi'
        newname2 = 'sfip2'
        par =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,1]
        phiend =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,2]
        M =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,3]
        dtphi =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,4]
    elif model == 'sfi4':
        newname1 = 'sfi'
        newname2 = 'sfip4'
        par =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,1]
        phiend =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,2]
        M =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,3]
        dtphi =  np.loadtxt('aspic/src/' + newname1 + '/' + newname2 + '_true.dat')[:,4]    
    else:
        par =  np.loadtxt('aspic/src/' + model + '/' + model + '_true.dat')[:,0]
        phiend =  np.loadtxt('aspic/src/' + model + '/' + model + '_true.dat')[:,1]
        M =  np.loadtxt('aspic/src/' + model + '/' + model + '_true.dat')[:,2]
        dtphi =  np.loadtxt('aspic/src/' + model + '/' + model + '_true.dat')[:,3]
    
    return par, phiend, M, dtphi

def find_phiend(par,M,phi,V,V1,i):
    par_val, phiend, M_scale, dtphi = extract_data(V1)
    phi_guess = phiend[i]
    func = scipy.interpolate.interp1d(phi,eps1(par,M,phi,V) - 1, fill_value = 'extrapolate')
    roots = fsolve(func, x0 = [phi_guess])
    real_root = []
    for i in range(len(roots)):
        if abs(roots[i].imag)<1e-5:
            real_root.append(roots[i].real)
    return real_root

def calculate_data():
    names = ['rchi','lfi','gmlfi','rcmi','rcqi','ni','esi','kmii','hf1i','cwi','rpi','dwi','mhi','rgi','gmssmi','gripi','ai','cnai','cnbi','osti','wri','sfi1','sfi2','sfi4']
    #names = ['rgi','kmii']
    potential = Potentials()
    
    for j in range(len(names)):

        par, phiend, M, dtphi = extract_data(names[j])  
        print(phiend)
        phi = np.linspace(min(phiend),max(phiend),10000)
        phiend_calc = []
                
        for i in range(len(par)): 
            V_attr = getattr(potential, str('V_')+str(names[j]).upper())
            phiend_calc.append(find_phiend(par[i],M[i],phi,V_attr,names[j],i)[0])
        plt.scatter(par,phiend)
        plt.loglog(par,phiend_calc)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$\phi_{\rm ini}$')
        plt.xlabel(r'$p$')
        plt.xlim(min(par),max(par))
        plt.ylim(min(phiend),max(phiend))
        plt.grid()
        plt.savefig('aspic/src/'+str(names[j])+'.pdf', bbox_inches='tight')
    
def main():
    calculate_data()
    
main()