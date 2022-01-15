import numpy as np

#### Get nomenclature from Introduction to Chemical Engineering Thermodynamics by Joseph Mauk Smith, Hendrick C. Van Ness, Michael M. Abbott, Mark Thomas Swihart 

class UNIFAC:
    def __init__(self):    
        self.x = np.array([[0.2, 0.8]]) # mol fractions
        self.T = np.array([[330]])  # K
        # Frequency of UNIFAC groups : each row denotes the subgroup and each column denotes the component
        self.nu = np.array([[2, 0],
                            [2, 0],
                            [1, 0],
                            [0, 1]])
        # R values
        self.R = np.array([0.9011, 0.6744, 1.6764, 3.1680])
        # Q values
        self.Q = np.array([0.8480, 0.5400, 1.4200, 2.4840])
        # a_mn values (energy contributions in residual)
        self.a = np.array([[0.0000, 0.0000, 232.10, 354.55],
                           [0.0000, 0.0000, 232.10, 354.55],
                           [114.80, 114.80, 0.0000, 202.30],
                           [-25.31, -25.31, -146.3, 0.0000]])
        self.r = np.matmul(self.R, self.nu)
        self.q = np.matmul(self.Q, self.nu)

    def get_gammaC(self):
        # Get the combinatorial part of activity coefficient
        # J = ri / sum(rj xj)
        J = np.zeros((len(self.x), len(self.x[0])))
        for i in range(len(self.x)):
            J[i] = self.r / np.dot(self.x[i], self.r)
        
        # L = qi / sum(qj xj)
        L = np.zeros((len(self.x), len(self.x[0])))
        for i in range(len(self.x)):
            L[i] = self.q / np.dot(self.x[i], self.q)

        lngammaC = 1 - J + np.log(J) - 5 * self.q * (1 - J / L + np.log(J / L))
        return np.exp(lngammaC)

    def get_gammaR(self):
        # Get the residual part of activity coefficient
        e = np.zeros(self.nu.transpose().shape)
        for i in range(e.shape[0]):
            e[i] = self.nu.transpose()[i] * self.Q / self.q[i]
        e = e.transpose()

        tau = np.exp(-self.a / self.T)
        beta = np.matmul(e.transpose(), tau)
        theta = np.zeros((len(self.x), len(self.nu)))
        for i in range(len(self.x)):
            for j in range(len(self.nu)):
                theta[i][j] = np.sum(self.x[i] * self.q * e[j, :]) / np.dot(self.x[i], self.q)

        s = np.matmul(theta, tau)
        lngammaR = np.zeros((len(self.x), len(self.x[0])))
        for i in range(len(self.x)):
            lngammaR[i] = self.q * (1 -
                                    (np.sum((theta[i, :] * beta / s[i, :]).transpose() -
                                            np.log(beta / s[i, :]).transpose() * e, axis=0))
                                    )
        return np.exp(lngammaR)

    def get_gamma(self):
        # Get the activity coefficent
        gammaC = self.get_gammaC()
        gammaR = self.get_gammaR()
        return gammaC * gammaR


# unifac = UNIFAC()
# print(unifac.get_gamma())
