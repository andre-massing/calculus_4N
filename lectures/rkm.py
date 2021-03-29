import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt


class EmbeddedExplicitRungeKutta:
    def __init__(self, a, b, c,  bhat=None, order=None):
        self.a = a
        self.b = b
        self.c = c
        self.bhat = bhat
        self.order = order

    def __call__(self, y0, t0, T, f, Nmax, tol=1e-3):
        # Extract Butcher table
        a, b, c, bhat, order = self.a, self.b, self.c, self.bhat, self.order

        # pessisimistic factor used in the new time-step computation 
        fac = 0.8     
        # some more advanced parameters for better controlling of time-step choice
        eps = 1e-15   # machine precision
        facmax = 5.0  # Maxima
        facmin = 0.1
        
        err  = 0
        
        # Stages
        s = len(b)
        ks = [np.zeros_like(y0, dtype=np.double) for s in range(s)]

        # Start time-stepping
        ys = [y0]
        ts = [t0]
        
        # Simple choice for initial time step
        dt = (T - t0)/Nmax
        # Counting steps
        N = 0
        N_rej = 0
        
        while(ts[-1] < T and N < Nmax):
            t, y = ts[-1], ys[-1]
            N += 1

            # Compute stages derivatives k_j
            for j in range(s):
                t_j = t + c[j]*dt
                dY_j = np.zeros_like(y, dtype=np.double)
                for l in range(j):
                    dY_j += a[j,l]*ks[l]

                ks[j] = f(t_j, y + dt*dY_j)
                
            # Compute next time-step
            dy = np.zeros_like(y, dtype=np.double)
            for j in range(s):
                dy += b[j]*ks[j]

            if bhat is None:
                ys.append(y + dt*dy)
                ts.append(t + dt)
            else:
                dyhat = np.zeros_like(y, dtype=np.double)
                for j in range(s):
                    dyhat += bhat[j]*ks[j]

                # Error estimate
                err = dt*norm(dy - dyhat)
                # ALTERNATIVE: more robust
                #err = max(dt*norm(dy - dyhat), norm(y)*eps)

                # Accept time-step
                if err <= tol:
                    ys.append(y + dt*dyhat)
                    ts.append(t + dt)
                else:
                    #print(f"Step is rejected at t = {t} with err = {err}")
                    N_rej += 1
                
                # Compute New step size
                dt = fac*(tol/err)**(1/(order+1))*dt
                # MORE ROBUST ALTERNATIVE
                #dt = min(dt*min(facmax, max(facmin, fac*(tol/err)**(1/(order+1)))),abs(T-ts[-1]))
        
        print(f"Finishing time-stepping reaching t = {ts[-1]} with final time T = {T}")
        print(f"Used {N} steps out of {Nmax} with {N_rej} being rejected")
          
        return (np.array(ts), np.array(ys))