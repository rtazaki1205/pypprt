import numpy as np
from tqdm import tqdm

# Monte Carlo 
# Code unit: Radiation moment H(0)=1.
def mcrt(Nphoto,taumax,Nmu,Nz):
    # random seed
    np.random.seed(123)

    # cell walls 
    mu_i = np.linspace(0, 1, Nmu+1)
    tau = np.linspace(0,taumax,Nz+1)

    # cell centers of mu grids
    mu_c = np.zeros(Nmu)
    for idir in range(Nmu):
        mu_c[idir] = 0.5*(mu_i[idir]+mu_i[idir+1])    
    
    # cell centers of tau grids
    tau_c = np.zeros(Nz)
    for iz in range(Nz):
        tau_c[Nz-iz-1] = 0.5 * (tau[iz]+tau[iz+1])

    # preparing arrays
    INTS = np.zeros(Nmu)
    ERR = np.zeros(Nmu)
    MJ = np.zeros(Nz+1)
    MH = np.zeros(Nz+1)
    MK = np.zeros(Nz+1)
    MJC = np.zeros(Nz)
    MHC = np.zeros(Nz)
    MKC = np.zeros(Nz)    
    
    Nout = 0
    for i in tqdm(range(Nphoto)):
        # drawing two random number
        xi=np.random.rand(2)
        cost = np.sqrt(xi[0])
        phi = 2.0*np.pi*xi[1]
        # initial location 
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        z1 = 0.0
        # iteration counter
        icount = 0
        while z1 <= 1:
            # storing triagonal functions
            sint = np.sqrt(1.0-cost*cost)
            sinp = np.sin(phi)
            cosp = np.cos(phi)
            # drawing another random number & determine the travel distance
            xi=np.random.rand()
            tau = -np.log(xi)
            s = tau / taumax
            # update location
            x1 = x0 + s * sint * cosp
            y1 = y0 + s * sint * sinp
            z1 = z0 + s * cost
            # check cell crossing
            j0=int(Nz*z0)
            j1=int(Nz*z1)   
            # if cell crossing happens, updating radiation moments
            if j0 != j1 or z0 <= 0.0 or z1 <= 0.0:
                if cost > 0.0:
                    if z0 > 0.0:
                        j0 += 1
                    j1 = min(j1+1,Nz+1)             
                elif cost < 0.0:
                    if z1 <= 0.0:
                        j1 = -1            
                sj=int(np.sign(cost))
                for j in range(j0,j1,sj):
                    MJ[j] += 1.0/abs(cost)
                    MH[j] += cost/abs(cost)
                    MK[j] += abs(cost)        
                    
            #  preparing for a next move of the photon packet
            if z1 < 0.0:
                # When it hits the ground, it is re-lauched again from the origin
                x0 = 0.0
                y0 = 0.0
                z0 = 0.0
                # reset the scattering counter
                icount = 0
                # lambertian emission
                xi=np.random.rand(2)
                cost = np.sqrt(xi[0])
                phi = 2.0*np.pi*xi[1]
            elif 0 <= z1 <= 1:
                # accept the next location
                x0 = x1
                y0 = y1
                z0 = z1
                # isotropic scattering
                xi=np.random.rand(2)
                cost = 2.0*xi[0]-1.0
                phi = 2.0*np.pi*xi[1]
                # count up the counter
                icount = icount + 1
         
        # storing the angle of the escaping photon packet
        idir = int(Nmu*cost)
        INTS[idir] += 1
        ERR[idir] += 1
        Nout += 1
        
    if Nout/Nphoto != 1.0:
        print('Strange! There are missing photons...')
        exit()
        
    for idir in range(Nmu):
        INTS[idir] = INTS[idir]*2.0*Nmu/mu_c[idir]/Nphoto
        if ERR[idir] != 0:
            ERR[idir]  = INTS[idir] / np.sqrt(ERR[idir])

    # compute radiation moments at the vertical cell center
    for iz in range(Nz):
        MJC[iz] = 0.5 * (MJ[iz]+MJ[iz+1]) / Nphoto
        MHC[iz] = 0.5 * (MH[iz]+MH[iz+1]) / Nphoto
        MKC[iz] = 0.5 * (MK[iz]+MK[iz+1]) / Nphoto
    
    return tau_c,mu_c,INTS,ERR,MJC,MHC,MKC


# Variable Eddington Method
def vef(taumax,Nmu,Nz):
    tau_i = np.linspace(0,taumax,Nz+1)
    dtau  = abs(tau_i[1]-tau_i[0])
    mu_min=dtau/30.0
    mu    = np.linspace(mu_min,1,Nmu+1) #mu=0 is not allowed.
    dmu = abs(mu[0]-mu[1])
    tau_c = np.zeros(Nz)
    J0 = np.zeros(Nz)
    for i in range(Nz):
        tau_c[i] = 0.5*(tau_i[i+1]+tau_i[i])
        
    J_init = (2.0+3.0*tau_c)
    for i in range(Nz):
        J0[i] = J_init[i]
        
    # Intensity, defined on the grid wall
    Iup = np.zeros((Nz+1,Nmu+1))
    Idown = np.zeros((Nz+1,Nmu+1))
    # moment at grid wall and center
    Ji = np.zeros(Nz+1)
    Hi = np.zeros(Nz+1)
    Ki = np.zeros(Nz+1)
    fc = np.zeros(Nz)
    Jc = np.zeros(Nz)
    Hc = np.zeros(Nz)
    Kc = np.zeros(Nz)
    itrmax = 10
    eps = 1.e-8

    for i in range(Nz):
        fc[i] = 1.0/3.0

    for itr in range(itrmax):
        Iup[:,:]=0.0
        Idown[:,:]=0.0
        # integrating the formal solution from top to bottom
        for j in range(Nmu+1):
            Idown[0,j]= 0.0
            for i in range(1,Nz+1):
                taueff = dtau/abs(mu[j])
                Idown[i,j]=Idown[i-1,j]*np.exp(-taueff)+J0[i-1]*(1.0-np.exp(-taueff))
            
        # measure the moment H at the ground surface
        Hi[Nz]=0.0
        for j in range(Nmu+1):
            Hi[Nz] += 0.5*Idown[Nz,j]*dmu*mu[j]
            
        # integrating the formal solution from bottom to top 
        for j in range(Nmu+1):
            Iup[Nz,j] = 4.0 + Hi[Nz]*4.0
            for i in range(1,Nz+1):
                taueff = dtau/abs(mu[j])
                Iup[Nz-i,j]=Iup[Nz-i+1,j]*np.exp(-taueff)+J0[Nz-i]*(1.0-np.exp(-taueff))
            
        # compute radiation moments at cell boundaries
        Ji[:]=0.0
        Hi[:]=0.0
        Ki[:]=0.0
        for i in range(Nz+1):
            for j in range(Nmu+1):
                Ji[i] += 0.5*(Iup[i,j]+Idown[i,j])*dmu
                Hi[i] += 0.5*(Iup[i,j]-Idown[i,j])*dmu*mu[j]
                Ki[i] += 0.5*(Iup[i,j]+Idown[i,j])*dmu*mu[j]*mu[j]  
        
        # compute radiation moment at cell center
        errmax = 0.0
        err=0.0
        for i in range(Nz):
            Jc[i] = 0.5*(Ji[i+1]+Ji[i])
            Hc[i] = 0.5*(Hi[i+1]+Hi[i])
            Kc[i] = 0.5*(Ki[i+1]+Ki[i])            
            err = abs(fc[i]-Kc[i]/Jc[i])/abs(Kc[i]/Jc[i])
        if err > errmax:
            errmax = err      
        print (itr,'errmax=',errmax)
        if errmax < eps:
            break
        else:
            # update the Eddington factor
            fc[i] = Kc[i]/Jc[i]
            # compute new mean intensity from new eddington factor
            for i in range(Nz):
                J0[i] = (tau_c[i]+2.0*fc[i])/fc[i]

    return tau_c,Jc,Hc,Kc
