import numpy as np
import sys
from .grid import Grid
from .math import tdma_v3_x, tdma_v3_y


class Solver(Grid):

    def __init__(self, NI, NJ, it, vtk, conditions):

        Grid.__init__(self, NI, NJ, it, vtk, conditions)
        super().__init__(NI, NJ, it, vtk, conditions)

        self.faceJU = np.zeros([self.NI+2, self.NJ+2])          # U face velocity for unregular face
        self.PU     = np.zeros([self.NI+2, self.NJ+2])          # U momentum source term
        self.PV     = np.zeros([self.NI+2, self.NJ+2])          # V momentum source term
        self.error  = np.zeros([self.NI+2, self.NJ+2])          # mass imbalance error

        self.ssum   = 0
                              
        self.link = {                                          
            'E': np.zeros([self.NI+2, self.NJ+2]),
            'W': np.zeros([self.NI+2, self.NJ+2]),
            'N': np.zeros([self.NI+2, self.NJ+2]),
            'S': np.zeros([self.NI+2, self.NJ+2]),
            'P': np.zeros([self.NI+2, self.NJ+2])
        }
        self.p_link = {                                         
            'E': np.zeros([self.NI+2, self.NJ+2]),
            'W': np.zeros([self.NI+2, self.NJ+2]),
            'N': np.zeros([self.NI+2, self.NJ+2]),
            'S': np.zeros([self.NI+2, self.NJ+2]),
            'P': np.zeros([self.NI+2, self.NJ+2])
        }
    

    def solve(self):
        '''
        This method is the main method of Solver
        '''
        vtk_iter = self.vtk
        FREQ = 100
        rU, rV, rP = self.conditions['underrelaxation']

        for outer_iter in range(self.iterations):
            # Monitoring conditions
            if outer_iter % FREQ == 0:
                
                if outer_iter % self.vtk == 0:
                    self.vtk += vtk_iter
                if self.convergence(outer_iter):
                    break

            # Calculations
            self.link_coefficients()            # Calculate the link coefficients
            self.boundary_conditions()          # Update links with boundary conditions
            self.simple(rU, rV, rP)             # Main algorithm 
                        
        print(f'\n>>> System solved successfully\n')


    def link_coefficients(self):
        '''
        This method calculates D and F for all faces
        and uses its values to obtain the link coefficients
        for all faces and P
        '''
        I       = slice(1, self.NI+1)
        J       = slice(1, self.NJ+1)
        zero    = np.zeros([self.NI, self.NJ])

        diff = {                                      
            'E': np.zeros([self.NI+2, self.NJ+2]),
            'W': np.zeros([self.NI+2, self.NJ+2]),
            'N': np.zeros([self.NI+2, self.NJ+2]),
            'S': np.zeros([self.NI+2, self.NJ+2])
        }
        conv = {                                      
            'E': np.zeros([self.NI+2, self.NJ+2]),
            'W': np.zeros([self.NI+2, self.NJ+2]),
            'N': np.zeros([self.NI+2, self.NJ+2]),
            'S': np.zeros([self.NI+2, self.NJ+2])
        }
        for i in range(1, self.NI+1):
            for j in range(1, self.NJ+1):
                # West
                if j == 1:
                    diff['W'][i, j] = 0
                    conv['W'][i, j] = 0
                else:
                    diff['W'][i, j] = self.visc[i, j] * self.area['W'][i, j] / self.dist['W'][i, j]
                    conv['W'][i, j] = self.dsty[i, j] * self.area['W'][i, j] * self.U_face[i, j-1]
                # East
                if j == self.NJ:
                    diff['E'][i, self.NJ] = 0
                    conv['E'][i, self.NJ] = 0
                else:
                    diff['E'][i, j] = self.visc[i, j] * self.area['E'][i, j] / self.dist['E'][i, j]
                    conv['E'][i, j] = self.dsty[i, j] * self.area['E'][i, j] * self.U_face[i, j]
                # North
                if i == 1:
                    diff['N'][i, j] = 0
                    conv['N'][i, j] = 0
                else:
                    diff['N'][i, j] = self.visc[i, j] * self.area['N'][i, j] / self.dist['N'][i, j]
                    conv['N'][i, j] = (self.dsty[i, j] * self.area['N'][i, j] * self.V_face[i-1, j]) + (self.dsty[i, j] * (self.Y_face[i-1, j] - self.Y_face[i-1, j-1]) * self.faceJU[i-1, j])
                # South
                if i == self.NI:
                    diff['S'][i, j] = 0
                    conv['S'][i, j] = 0
                else:
                    diff['S'][i, j] = self.visc[i, j] * self.area['S'][i, j] / self.dist['S'][i, j]
                    conv['S'][i, j] = (self.dsty[i, j] * self.area['S'][i, j] * self.V_face[i, j]) + (self.dsty[i, j] * (self.Y_face[i, j] - self.Y_face[i, j-1]) * self.faceJU[i, j])
                
        # Hybrid scheme for the link coefficients
        self.link['W'][I, J] = np.maximum.reduce([conv['W'][I, J], diff['W'][I, J] + conv['W'][I, J]/2, zero])
        self.link['E'][I, J] = np.maximum.reduce([- conv['E'][I, J], diff['E'][I, J] - conv['E'][I, J]/2, zero])
        self.link['N'][I, J] = np.maximum.reduce([- conv['N'][I, J], diff['N'][I, J] - conv['N'][I, J]/2, zero])
        self.link['S'][I, J] = np.maximum.reduce([conv['S'][I, J], diff['S'][I, J] + conv['S'][I, J]/2, zero])
        self.link['P'][I, J] = (self.link['W'][I, J] + self.link['E'][I, J] + self.link['N'][I, J] + self.link['S'][I, J] 
                                + conv['E'][I, J] - conv['W'][I, J] + conv['S'][I, J] - conv['N'][I, J])

        # Zero pivot check     
        small_p_links = self.link['P'][I, J] < 1e-12
        if np.any(small_p_links):
            print('\n[SOLVER] SMALL NODE LINK - SYSTEM WILL CRASH')
            print(small_p_links)
            print('Exiting ...')
            sys.exit()


    def boundary_conditions(self):
        '''
        This method fills boundary conditions
        for the link coefficients
        '''
        for i in range(1, self.NI+1):
            for j in range(1, self.NJ+1):

                Ax, Ay      = lambda i, j: self.x_face_area(i, j), lambda i, j: self.y_face_area(i, j)
                Axx, Ayy    = lambda i, j: self.incline_x_face(i, j), lambda i, j: self.incline_y_face(i, j)

                # X momentum
                if j == 1:
                    # inlet pressure (there might be a problem here - rewrite PU expressions which refer to convective and diffusive)
                    self.P[i, 0] = 1.5 * self.P[i, 1] - 0.5 * self.P[i, 2]
                    self.PU[i, j] = (Ay(i, j) * (self.P[i, j-1] - 0.5 * (self.P[i, j] + self.P[i, j+1]))
                                    + (Ay(i, j) * self.dsty[i, j-1] * self.U[i, j-1] * self.U[i, j-1]) 
                                    + ((Ay(i, j) * 8 * self.visc[i, j-1] * self.U[i, j-1]) / (3 * self.dist['W'][i, j])))

                    # inlet - taylor expansion at two points
                    self.link['E'][i, j] += (self.visc[i, j] * (self.Y_face[i, j] - self.Y_face[i-1, j]) / (3 * Ax(i, j)))
                    self.link['P'][i, j] += (3 * self.visc[i, j] * (self.Y_face[i, j] - self.Y_face[i-1, j]) / (Ax(i, j)))

                elif j == self.NJ:
                    # outlet pressure
                    self.PU[i, j] = Ay(i, j) * (0.5 * (self.P[i, j] + self.P[i, j-1]) - self.P[i, j+1])
                    
                    # links
                    self.link['W'][i, j] += 0.5 * Ay(i, j) * self.dsty[i, j] * self.U[i, j]
                    self.link['P'][i, j] += 1.5 * Ay(i, j) * self.dsty[i, j] * self.U[i, j]
                
                else:
                    # center of the domain
                    self.PU[i, j] = 0.5 * Ay(i, j) * (self.P[i, j-1] - self.P[i, j+1])
                    
                # Y momentum
                if i == 1:
                    # source term north boundary
                    self.P[0, j] = 1.5 * self.P[1, j] - 0.5 * self.P[2, j]
                    self.PV[i, j] = Ax(i, j) * (self.P[i-1, j] - (0.5 * (self.P[i, j] + self.P[i+1, j])))
                    self.PU[i, j] += Ayy(i, j) * (self.P[i-1, j] - (0.5 * (self.P[i, j] + self.P[i+1, j])))

                    # north inlet (there might also be a problem here)
                    if j > 6 and j < 16:
                        self.PV[i, j] += (Ax(i, j) * self.dsty[i-1, j] * self.V[i-1, j] * self.V[i-1, j]) + ((Ax(i, j) * 8 * self.visc[i-1, j] * self.V[i-1, j]) / (3 * Ay(i, j)))

                    self.link['S'][i, j] += (self.visc[i, j] * (self.X_face[i, j] - self.X_face[i, j-1]) / (3 * Ay(i, j)))
                    self.link['P'][i, j] += (3 * self.visc[i, j] * (self.X_face[i, j] - self.X_face[i, j-1]) / (Ay(i, j)))
                
                elif i == self.NI:
                    # source term south boundary
                    self.P[i+1, j] = 1.5 * self.P[i, j] - 0.5 * self.P[i-1, j]
                    self.PV[i, j] = Ax(i, j) * (0.5 * (self.P[i, j] + self.P[i-1, j]) - self.P[i+1, j])
                    self.PU[i, j] += Ayy(i, j) * (0.5 * (self.P[i, j] + self.P[i-1, j]) - self.P[i+1, j])

                    # south inlet
                    if j > 6 and j < 16:
                        self.PV[i, j] += (Ax(i, j) * self.dsty[i+1, j] * self.V[i+1, j] * self.V[i+1, j]) + ((Ax(i, j) * 8 * self.visc[i+1, j] * self.V[i+1, j]) / (3 * Ay(i, j)))

                    # links
                    self.link['N'][i, j] += (self.visc[i, j] * (self.X_face[i, j] - self.X_face[i, j-1]) / (3 * Ay(i, j)))
                    self.link['P'][i, j] += (3 * self.visc[i, j] * (self.X_face[i, j] - self.X_face[i, j-1]) / (Ay(i, j)))
                else:
                    # center of the domain
                    self.PU[i, j] += 0.5 * Ayy(i, j) * (self.P[i-1, j] - self.P[i+1, j])
                    self.PV[i, j] = 0.5 * Ax(i, j) * (self.P[i-1, j] - self.P[i+1, j])


    def simple(self, rU, rV, rP):
        '''
        This method calls the SIMPLE algorithm to
        calculate face velocities
        '''
        Pcor = np.zeros([self.NI+2, self.NJ+2])

        # Linear system solution
        self.U = tdma_v3_x(self.NI, self.NJ, self.link, self.U, self.PU, idt='U')
        self.V = tdma_v3_y(self.NI, self.NJ, self.link, self.V, self.PV, idt='V')

        # Updating face velocities
        self.new_face_velocity(rU, rV)

        # Pressure correction
        self.ssum, lim = self.mass_imbalance()
        self.pressure_correction_coefficients(rU, rP)
        Pcor = tdma_v3_x(self.NI, self.NJ, self.p_link, Pcor, self.error, idt='P')
        self.P += rP * Pcor

        # Velocity correction
        U_old = self.U.copy()
        V_old = self.V.copy()
        self.velocity_correction(rU, rP, Pcor)

        
    def new_face_velocity(self, rU, rV):
        '''
        This method calculates and directly updates 
        new values for the face velocities
        '''
        # Updating U face velocities
        for i in range(1, self.NI+1):
            for j in range(1, self.NJ):
                self.U_face[i, j] = ((1.0 - rU) * self.U_face[i, j] 
                                    + rU * 0.5 * (self.U[i, j] + self.U[i, j+1])
                                    - rU * 0.5 * (self.Y_face[i, j] - self.Y_face[i-1, j])
                                        * (self.P[i, j+1] - self.P[i, j])
                                        * ((1 / self.link['P'][i, j+1]) + (1 / self.link['P'][i, j])))
                    
            self.U[i, self.NJ+1] = (1.5 * self.U[i, self.NJ]) - (0.5 * self.U[i, self.NJ-1])
            self.V[i, self.NJ+1] = self.V[i, self.NJ]
            self.U_face[i, self.NJ] = self.U[i, self.NJ+1]
        
        # Updating V face velocities
        for j in range(1, self.NJ+1):
            for i in range(1, self.NI):

                self.faceJU[i, j] = 0.5 * (self.U[i, j] + self.U[i+1, j])

                self.V_face[i, j] = (((1.0 - rV) * self.V_face[i, j]) 
                                    + rV * 0.5 * (self.V[i, j] + self.V[i+1, j])
                                    - rV * 0.5 * (self.X_face[i, j] - self.X_face[i, j-1])
                                        * (self.P[i+1, j] - self.P[i, j])
                                        * ((1 / self.link['P'][i+1, j]) + (1 / self.link['P'][i, j])))
                

    def mass_imbalance(self):
        '''
        This method calculates the mass imbalance
        and returns the calculated total error
        '''
        ssum = 0         
        lim = 1.1e-24

        for i in range(1, self.NI+1):
            for j in range(1, self.NJ+1):

                werror, eerror, nerror, serror = 0, 0, 0, 0
                
                nerror = ((self.dsty[i, j] * self.V_face[i-1, j] * (self.X_face[i-1, j] - self.X_face[i-1, j-1]))
                            + (self.dsty[i, j] * self.faceJU[i-1, j] * (self.Y_face[i-1, j] - self.Y_face[i-1, j-1])))
                serror = ((self.dsty[i, j] * self.V_face[i, j] * (self.X_face[i, j] - self.X_face[i, j-1]))
                            + (self.dsty[i, j] * self.faceJU[i, j] * (self.Y_face[i, j] - self.Y_face[i, j-1])))
                
                if j == self.NJ:
                    eerror = (1.5 * self.dsty[i, j] * self.U[i, j] * (self.Y_face[i, j] - self.Y_face[i-1, j]))
                    werror = (1.5 * self.dsty[i, j] * self.U_face[i, j-1] * (self.Y_face[i, j-1] - self.Y_face[i-1, j-1]))
                else:
                    eerror = (self.dsty[i, j] * self.U_face[i, j] * (self.Y_face[i, j] - self.Y_face[i-1, j]))
                    werror = (self.dsty[i, j] * self.U_face[i, j-1] * (self.Y_face[i, j-1] - self.Y_face[i-1, j-1]))

                self.error[i, j] = werror - eerror + nerror - serror

                ssum += abs(self.error[i, j])
                lim = max(lim, abs(self.error[i, j]))
        
        return ssum, lim

    def pressure_correction_coefficients(self, rU, rV):
        '''
        This method calculates the link coefficients
        for the pressure correction
        '''
        for i in range(1, self.NI+1):
            for j in range(1, self.NJ):
                
                # North wall
                if i == 1:
                    self.p_link['N'][i, j] = 0
                else:
                    self.p_link['N'][i, j] = (0.5 * self.dsty[i, j] * (self.area['N'][i, j])**2 
                                            * (rV / (1 - rV)) * ((1 / self.link['P'][i, j]) + (1 / self.link['P'][i-1, j])))
                
                # South wall
                if i == self.NI:
                    self.p_link['S'][i, j] = 0
                else:
                    self.p_link['S'][i, j] = (0.5 * self.dsty[i, j] * (self.x_face_area(i, j))**2 
                                            * (rV / (1 - rV)) * ((1 / self.link['P'][i+1, j]) + (1 / self.link['P'][i, j])))
                
                # West boundary - INLET
                if j == 1:
                    self.p_link['W'][i, j] = 0
                else:
                    self.p_link['W'][i, j] = 0.5 * self.dsty[i, j] * (self.area['W'][i, j])**2 \
                                            * (rU / (1 - rU)) * ((1 / self.link['P'][i, j]) + (1 / self.link['P'][i, j-1]))
                    
                self.p_link['E'][i, j] = 0.5 * self.dsty[i, j] * (self.area['E'][i, j])**2 \
                                            * (rU / (1 - rU)) * ((1 / self.link['P'][i, j]) + (1 / self.link['P'][i, j+1]))
                
                self.p_link['P'][i, j] = self.p_link['W'][i, j] + self.p_link['E'][i, j] + self.p_link['N'][i, j] + self.p_link['S'][i, j]
        
        # Dedicated loop for the outlet
        for i in range(1, self.NI+1):
            
            # East boundary - OUTLET
            if i == 1:
                self.p_link['N'][i, self.NJ] = 0
            else:
                self.p_link['N'][i, self.NJ] =  (0.5 * self.dsty[i, self.NJ] * (self.area['N'][i, self.NJ])**2                               
                                                * (rV / (1 - rV)) * ((1 / self.link['P'][i, self.NJ]) + (1 / self.link['P'][i-1, self.NJ])))
            if i == self.NI:
                self.p_link['S'][i, self.NJ] = 0
            else:
                self.p_link['S'][i, self.NJ] = 0.5 * self.dsty[i, self.NJ] * (self.x_face_area(i, j))**2                                \
                                            * (rV / (1 - rV)) * ((1 / self.link['P'][i+1, self.NJ]) + (1 / self.link['P'][i, self.NJ]))
            self.p_link['E'][i, self.NJ] = 0
            self.p_link['W'][i, self.NJ] = 1.5 * self.dsty[i, self.NJ] * (self.area['W'][i, self.NJ])**2                                    \
                                                * (rU / (1 - rU)) * ((1 / self.link['P'][i, self.NJ]) + (1 / self.link['P'][i, self.NJ-1])) \
                                            - 1.5 * self.dsty[i, self.NJ] * (self.area['W'][i, self.NJ])**2                                 \
                                                * (rU / (1 - rU)) * (1 / self.link['P'][i, self.NJ])
            
            self.p_link['P'][i, self.NJ] = 1.5 * self.dsty[i, self.NJ] * (self.area['W'][i, self.NJ])**2                                    \
                                                * (rU / (1 - rU)) * ((1 / self.link['P'][i, self.NJ]) + (1 / self.link['P'][i, self.NJ-1])) \
                                            + 1.5 * self.dsty[i, self.NJ] * (self.area['W'][i, self.NJ])**2                                 \
                                                * (rU / (1 - rU)) * (1 / self.link['P'][i, self.NJ])                                  \
                                            + self.p_link['S'][i, self.NJ]                                                      \
                                            + self.p_link['N'][i, self.NJ]


    def velocity_correction(self, rU, rV, Pcor):
        '''
        This method makes the final velocity
        corrections after the pressure correction
        and prepares the system for the next iteration
        '''
        # Node velocities
        for i in range(1, self.NI+1):
            for j in range(1, self.NJ+1):

                self.U[i, j] += (rU * self.y_face_area(i, j)) * (0.5 * (Pcor[i, j-1] - Pcor[i, j+1]) / self.link['P'][i, j])
                self.V[i, j] += (rV * self.x_face_area(i, j)) * (0.5 * (Pcor[i-1, j] - Pcor[i+1, j]) / self.link['P'][i, j])

        # Face U velocity
        for i in range(1, self.NI+1):
            for j in range(1, self.NJ):
                self.U_face[i, j] += rU * self.y_face_area(i, j) * 0.5 * (Pcor[i, j] - Pcor[i, j+1]) * (1 / self.link['P'][i, j+1] + 1 / self.link['P'][i, j])

        # Face V velocity
        for j in range(1, self.NJ+1):
            for i in range(1, self.NI):
                self.V_face[i, j] += rV * self.x_face_area(i, j) * 0.5 * (Pcor[i, j] - Pcor[i+1, j]) * (1 / self.link['P'][i+1, j] + 1 / self.link['P'][i, j])
                self.faceJU[i, j] = 0.5 * (self.U[i, j] + self.U[i+1, j])

        # Side inlets
        self.V[1, 6:18] = self.V[0, 6:18]
        self.V[self.NI, 6:18] = self.V[self.NI+1, 6:18]
        
        # Pressure corners
        self.P[0, 0] = self.P[1, 0]
        self.P[self.NI+1, 0] = self.P[self.NI, 0]
        self.P[0, self.NJ+1] = self.P[1, self.NJ+1]
        self.P[self.NI+1, self.NJ+1] = self.P[self.NI, self.NJ+1]


    def convergence(self, outer_iteration, r_thresh=2e-5, b_thresh=0.01):
        '''
        This method checks if the calculation has converged
        Criteria 1 -> differential mass balance
        Criteria 2 -> integral mass balance
        '''
        PATH = 'log.txt'

        residue = self.ssum
        in_out, inlet = self.global_mass_balance()
        balance = in_out / inlet

        print(f'[SOLVER] iteration {outer_iteration}\t residue {residue}\t balance {balance}')

        data = [self.U[20, 90], self.U[20, 180], 
                self.P[20, 90], self.P[20, 180], 
                self.ssum, balance]
        self.append_to_csv_file(PATH, data)


        if outer_iteration > 1:
            if residue < r_thresh or balance < b_thresh:
                print(f'\n[SOLVER] calculation converged after {outer_iteration} iterations')
                return True
        return False