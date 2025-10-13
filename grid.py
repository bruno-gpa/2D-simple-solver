import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.integrate import simpson


class Grid:

    def __init__(self, NI, NJ, iterations, vtk, conditions):
        self.NI = NJ
        self.NJ = NI
        self.iterations = int(iterations)
        self.vtk = 1
        self.conditions = conditions

        self.X       = np.zeros((self.NI+2, self.NJ+2))
        self.Y       = np.zeros((self.NI+2, self.NJ+2))
        self.X_face  = np.zeros((self.NI+2, self.NJ+2))
        self.Y_face  = np.zeros((self.NI+2, self.NJ+2))

        self.dsty   = None
        self.visc   = None
        self.P      = None
        self.U      = None
        self.V      = None
        self.U_face = None
        self.V_face = None

        self.area = {                                     
            'E': np.zeros([self.NI+2, self.NJ+2]),
            'W': np.zeros([self.NI+2, self.NJ+2]),
            'N': np.zeros([self.NI+2, self.NJ+2]),
            'S': np.zeros([self.NI+2, self.NJ+2])
        }
        self.dist = {                                     
            'E': np.zeros([self.NI+2, self.NJ+2]),
            'W': np.zeros([self.NI+2, self.NJ+2]),
            'N': np.zeros([self.NI+2, self.NJ+2]),
            'S': np.zeros([self.NI+2, self.NJ+2])
        }
    
    
    def build_grid(self):
        PATH = 'log.txt'
        header = ['U center [m/s]', 'U outlet [m/s]', 'P center [Pa]', 
                  'P outlet [Pa]', 'error', 'mass balance']
        self.append_to_csv_file(PATH, header)
        
        self.create_grid()
        self.calculate_area()
        self.guess()
        self.boundaries()
        self.build_faces()

        print('\n[GRID] i = ' + str(self.NI+2) + ' x j = ' + str(self.NJ+2))
        print('\n>>> Grid built successfully\n')


    def create_grid(self):
        with open('files/geom') as geom:

            for j, line in enumerate(geom):
                # get data from the file (i -> NI+1)
                xlow, ylow, xhigh, yhigh = map(float, line.rstrip().split('\t'))
                xlow, ylow, xhigh, yhigh = xlow/1000, ylow/1000, xhigh/1000, yhigh/1000
                dy = (yhigh - ylow) / self.NI

                self.X_face[0, j] = xlow
                self.Y_face[0, j] = ylow

                # fill the current wall cell with data from the previous cell + the delta
                for i in range(1, self.NI+1):
                    self.X_face[i, j] = self.X_face[i-1, j]
                    self.Y_face[i, j] = self.Y_face[i-1, j] + dy
                
                # fill the current node cell with data from the current and previous wall cells
                if j > 0:
                    for i in range(self.NI+2):
                        self.X[i, j] = self.X_face[i, j-1] + 0.5*(self.X_face[i, j] - self.X_face[i, j-1])
                        if i > 0:
                            self.Y[i, j] = self.Y_face[i-1, j] + 0.5*(self.Y_face[i, j] - self.Y_face[i-1, j])

                # boundary conditions
                self.X[:, self.NJ+1]  = self.X_face[:, self.NJ]
                self.X[0, :]          = self.X[1, :]
                self.X[self.NI+1, :]  = self.X[self.NI, :]
                
                self.Y[self.NI+1, :]  = self.Y_face[self.NI, :]
                self.Y[:, 0]          = self.Y[:, 1]
                self.Y[:, self.NJ+1]  = self.Y[:, self.NJ]


    def guess(self):

        dsty_a, dsty_b = self.conditions.get('density')
        visc_a, visc_b = self.conditions.get('viscosity')

        self.dsty    = np.ones([self.NI+2, self.NJ+2]) * 0.5 * (dsty_a + dsty_b)
        self.visc    = np.ones([self.NI+2, self.NJ+2]) * 0.5 * (visc_a + visc_b)

        self.P       = np.ones([self.NI+2, self.NJ+2]) * 0.0005 * 1000

        self.U       = np.zeros([self.NI+2, self.NJ+2])
        self.V       = np.zeros([self.NI+2, self.NJ+2])

        self.U_face  = np.zeros([self.NI+2, self.NJ+2])
        self.V_face  = np.zeros([self.NI+2, self.NJ+2])


    def boundaries(self):
        inlet = slice(6, 18)
        Uin, Vin, Pout = self.conditions.get('boundaries')

        # top and bottom walls
        self.U[0, :]                    = 0
        self.U[self.NI+1,  :]           = 0
        self.U_face[0, :]               = 0
        self.U_face[self.NI, :]         = 0

        self.V[0, :]                    = 0
        self.V[self.NI+1,  :]           = 0
        self.V_face[0, :]               = 0
        self.V_face[self.NI, :]         = 0
        
        # main inlet
        self.U[:, 0]                    = Uin
        self.U_face[:, 0]               = Uin

        self.V[:, 0]                    = 0
        self.V_face[:, 0]               = 0

        # outlet
        self.P[:, self.NJ+1]            = Pout
        
        # side inlet
        self.U[0, inlet]                 = 0
        self.U[self.NI+1, inlet]         = 0
        self.U_face[0, inlet]            = 0
        self.U_face[self.NI, inlet]      = 0

        self.V[0, inlet]                 = Vin
        self.V[self.NI+1, inlet]         = - Vin
        self.V_face[0, inlet]            = Vin
        self.V_face[self.NI, inlet]      = - Vin
    

    def build_faces(self):

        for i in range(1, self.NI+1):
            for j in range(1, self.NJ+1):
                
                self.area['W'][i, j] = 0.5*(self.Y_face[i, j] - self.Y_face[i-1, j] + self.Y_face[i, j-1] - self.Y_face[i-1, j-1])
                self.dist['W'][i, j] = self.X[i, j] - self.X[i, j-1]

                self.area['E'][i, j] = 0.5*(self.Y_face[i, j+1] - self.Y_face[i-1, j+1] + self.Y_face[i, j] - self.Y_face[i-1, j])
                self.dist['E'][i, j] = self.X[i, j+1] - self.X[i, j]

                self.area['N'][i, j] = 0.5*(self.X_face[i, j] - self.X_face[i, j-1] + self.X_face[i-1, j] - self.X_face[i-1, j-1])
                self.dist['N'][i, j] = self.Y[i, j] - self.Y[i-1, j]

                self.area['S'][i, j] = 0.5*(self.X_face[i+1, j] - self.X_face[i+1, j-1] + self.X_face[i, j] - self.X_face[i, j-1])
                self.dist['S'][i, j] = self.Y[i+1, j] - self.Y[i, j]


    def calculate_area(self):
        surf = np.zeros([self.NI+2, self.NJ+2])
        for i in range(self.NI):
            for j in range(self.NJ):
                surf[i, j] = 0.5*( ((self.X[i, j+1] - self.X[i, j]) * (self.Y[i+1, j+1] - self.Y[i, j+1])) \
                            + (self.X[i+1, j+1] - self.X[i+1, j]) * (self.Y[i+1, j] - self.Y[i, j]) )
                self.check_area(surf[i, j])
    

    def check_area(self, surf):
        if surf < 0:
            raise Exception('>>> Negative area encountered')
        elif surf == 0:
            raise Exception('>>> Zero area encountered')
    

    def x_face_area(self, i, j):
        return 0.5 * (self.X_face[i, j] - self.X_face[i, j-1] + self.X_face[i-1, j] - self.X_face[i-1, j-1])
        

    def y_face_area(self, i, j):
        return 0.5 * (self.Y_face[i, j] - self.Y_face[i-1, j] + self.Y_face[i, j-1] - self.Y_face[i-1, j-1])
    

    def incline_x_face(self, i, j):
        return 0.5 * (self.X_face[i, j] - self.X_face[i-1, j] + self.X_face[i, j-1] - self.X_face[i-1, j-1])


    def incline_y_face(self, i, j):
        return 0.5 * (self.Y_face[i, j] - self.Y_face[i, j-1] + self.Y_face[i-1, j] - self.Y_face[i-1, j-1])


    def show(self, property, name):
        print(name)
        print(property)
        print(np.shape(property))
    

    def plot_field(self, property, name, scale=50):
        fig, ax = plt.subplots(figsize=(10, 10*self.NJ/self.NI))
        im = ax.imshow(property, interpolation="quadric", cmap='Spectral', vmin=-scale, vmax=scale, aspect='equal')
        ax.invert_yaxis()

        x_ticks = np.linspace(0, property.shape[1] - 1, num=5, dtype=int)
        x_labels = np.linspace(0, self.X[0, -1], num=10, dtype=int)
        plt.xticks(x_ticks, x_labels)

        y_ticks = np.linspace(0, property.shape[0] - 1, num=10, dtype=int)
        y_labels = np.linspace(0, self.Y[[-1, 0]], num=5, dtype=int)
        plt.yticks(y_ticks, y_labels)

        ax.contour(property, levels=[0], colors='white')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title('Plot - ' + name)
        im_ratio = property.shape[0]/property.shape[1]
        plt.colorbar(im, fraction=0.047*im_ratio, pad=0.08)

        plt.show()


    def plot_hline(self, properties, y, name):
        for p in properties:
            line = p[y, :]
            plt.plot(np.arange(p.shape[1]), line)
        plt.xlabel('X nodes')
        plt.ylabel(name)
        plt.title('Line plot - ' + name)
        plt.show()
    

    def plot_cross(self, properties, x, name):
        for p in properties:
            line = p[:, x]
            y = np.arange(p.shape[0])
            plt.plot(y, line)
        plt.xlabel('Y nodes')
        plt.ylabel(name)
        plt.title('Cross section plot - ' + name)
        plt.show()


    def append_to_csv_file(self, file_path, values):
    
        new_line = " \t".join(map(str, values)) + "\n"

        with open(file_path, 'a') as f:
            f.write(new_line)
    

    def global_mass_balance(self):
        y_main_inlet = self.U[:, 0]
        x_main_inlet = self.Y[:, 0]

        y_side_inlet = 2 * self.V[0, :]
        x_side_inlet = self.X[0, :]

        y_outlet = self.U[:, self.NJ+1]

        inlet = simpson(y_main_inlet, x_main_inlet) + simpson(y_side_inlet, x_side_inlet)
        outlet = simpson(y_outlet, x_main_inlet)

        return inlet - outlet, inlet
