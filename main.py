import numpy as np
from modules import grid, solver


def main():
    PATH = 'files/flow_macro'

    with open(PATH) as flow:
        NI, NJ = map(int, flow.readline(-1).split(','))
        params = list(map(float, flow.readline(-1).split(',')))
        mass_transfer = list(map(float, flow.readline(-1).split(',')))
        boundaries = list(map(float, flow.readline(-1).split(',')))
        relax = list(map(float, flow.readline(-1).split(',')))
        it, vtk = map(float, flow.readline(-1).split(','))
    
    conditions =    {'viscosity': params[::2],
                    'density': params[1::2],
                    'mass transfer': mass_transfer,
                    'boundaries': boundaries,
                    'underrelaxation': relax}

    mysolver = solver.Solver(NI, NJ, it, vtk, conditions)
    mysolver.build_grid()
    mysolver.solve()

    mysolver.plot_field(mysolver.U, 'U vel', np.max(mysolver.U))
    mysolver.plot_field(mysolver.V, 'V vel', np.max(mysolver.V))
    mysolver.plot_field(mysolver.P, 'P', np.max(mysolver.P))
    mysolver.plot_hline([mysolver.U], 20, 'Velocities')
    mysolver.plot_cross([mysolver.U], 180, 'Velocities')


if __name__ == "__main__":
    main()

