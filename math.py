import numpy as np
import sys


def tdma_v3_y(NI, NJ, link, field, source, idt):
    '''
    This method implements the TDMA solver
    for tri-diagonal matrices
    V3 Y : column sweep solution for Y momentum
    '''
    phi = field.copy()
    
    for j in range(1, NJ+1):
        a = np.zeros(NI)            # lower diagonal (south links)
        b = np.zeros(NI)            # main diagonal (node links)
        c = np.zeros(NI)            # upper diagonal (north links)
        d = np.zeros(NI)            # sources

        section = np.arange(1, NI+1)
        b[:] = link['P'][section, j]
        a[:] = link['S'][section, j]
        c[:] = link['N'][section, j]
        d[:] = (link['W'][section, j] * phi[section, j-1]
                + link['E'][section, j] * phi[section, j+1] + source[section, j])

        d[0] += c[0] * phi[0, j]
        c[0] = 0
        d[-1] += a[-1] * phi[NI+1, j]
        a[-1] = 0

        try:
            solution = thomas_solver(-a, b, -c, d, j)
        except ZeroDivisionError:
            print('[MATH] Singular matrix encoutered - zero in main diagonal')
            print('Exiting ...')
            sys.exit()
        phi[1:-1, j] = solution
    return phi


def tdma_v3_x(NI, NJ, link, field, source, idt):
    '''
    This method implements the TDMA solver
    for tri-diagonal matrices
    V3 X : row sweep solution for X momentum and P correction
    '''
    phi = field.copy()
    
    for i in range(1, NI+1):
        a = np.zeros(NJ)            # lower diagonal (west links)
        b = np.zeros(NJ)            # main diagonal (node links)
        c = np.zeros(NJ)            # upper diagonal (east links)
        d = np.zeros(NJ)            # sources

        section = np.arange(1, NJ+1)
        b[:] = link['P'][i, section]
        a[:] = link['W'][i, section]
        c[:] = link['E'][i, section]
        d[:] = (link['N'][i, section] * phi[i-1, section]
                + link['S'][i, section] * phi[i+1, section] + source[i, section])

        d[0] += a[0] * phi[i, 0]
        a[0] = 0
        d[-1] += c[-1] * phi[i, NJ+1]
        c[-1] = 0

        try:
            solution = thomas_solver(-a, b, -c, d, i)
        except ZeroDivisionError:
            print('[MATH] Singular matrix encoutered - zero in main diagonal')
            print('Exiting ...')
            sys.exit()
        phi[i, 1:-1] = solution

    return phi


def thomas_solver(a, b, c, d, j, tiny=1e-14):

    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[0] = c[0] / b[0] if n > 1 else 0.0
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
            
        cp[i] = c[i] / denom if i < n-1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    
    x[-1] = dp[-1]

    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x
