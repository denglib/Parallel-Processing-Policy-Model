# Parallel-Processing-Policy-Model
the run code of analytical model for shuttle-based compact storage system under parallel processing policy

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as li
import itertools as it


class RealPM:

    def __init__(self, data_original):

        # define the parameters

        r = data_original[0]             # arrival rate for retrieval transactions
        s = data_original[1]             # arrival rate for storage transactions
        Ntot = data_original[2]          # total number of storage locations
        ratio = data_original[3]         # the depth/width ratio
        self.n_s = data_original[4]      # number of shuttles
        u_w = data_original[5]           # unit width
        u_d = data_original[6]           # unit depth
        t_t = data_original[7]           # constant time for the transfer car to pick up/release the load
        t_sh = data_original[8]          # constant time for shuttles to pick up/release the load
        v_t = data_original[9]           # the maximum velocity of the transfer car
        v_sh = data_original[10]         # the maximum velocity of shuttles
        a_t = data_original[11]          # acceleration/deceleration of the transfer car
        a_sh = data_original[12]         # acceleration/deceleration of shuttles
        self.la = r + s                  # total arrival rate for transactions
        n_c = int(np.ceil(np.sqrt(Ntot * ratio * u_w / (2 * u_d)))) # number of storage lanes at each side of cross-aisle
        n_l = int(np.ceil(Ntot / (2*n_c)))                          # number of storage columns at each side of cross-aisle

        # if estimate the performance of the real case in the "as is" scenario, then use:
        # n_c = 37
        # n_l = 47

        self.r = r
        self.s = s

        # probability of each transaction class and dwell point of resources
        p_s = s / (r + s)                                       # storage transactions
        p_r = r / (r + s)                                       # retrieval transactions
        s_iu = r / (r + s)                                      # shuttle dwells at l/u point
        s_c = s / (r + s)                                       # shuttle dwells at interior point
        c_iu = ((2*r**2+6*r*s+4*s**2)*n_l-r*s)/(4*(r+s)**2*n_l) # the transfer car dwells at l/u point
        c_c = ((2*r**2+2*r*s)*n_l+r*s)/(4*(r+s)**2*n_l)         # the transfer car dwells at interior point
        sa_po = 1 / (2*n_l)                                     # shuttle dwells at the same lane of the retrieval load
        di_po = (2*n_l - 1) / (2*n_l)                           # shuttle dwells at a different lane of the retrieval load

        # probability of each scenario
        self.p_1 = p_s*s_c*c_iu
        self.p_2 = p_s*s_c*c_c
        self.p_3 = p_s*s_iu*c_iu
        self.p_4 = p_s*s_iu*c_c
        self.p_5 = p_r*s_c*c_iu*sa_po
        self.p_6 = p_r*s_c*c_c*sa_po
        self.p_7 = 0.5*p_r*s_c*c_iu*di_po
        self.p_8 = 0.5*p_r*s_c*c_c*di_po
        self.p_9 = 0.5*p_r*s_iu*c_iu
        self.p_10 = 0.5*p_r*s_iu*c_c
        self.p_11 = c_iu*(self.p_7+self.p_8+self.p_9+self.p_10)
        self.p_12 = c_c*(self.p_7+self.p_8+self.p_9+self.p_10)

        def s_3(n_c, v_sh, u_d, t_sh, a_sh):

            distance_s_1 = 0
            for i in range(1, n_c+1):
                for j in range(1, n_c+1):
                    distance_s_1 += (abs(i-j)*u_d) / (n_c**2)
            if distance_s_1 > v_sh**2/a_sh:
                s_3_time_1 = 2*v_sh/a_sh+(distance_s_1-2*v_sh**2/(2*a_sh))/v_sh
            else:
                s_3_time_1 = 2*np.sqrt(distance_s_1/a_sh)

            distance_s_2 = ((n_c-1)*u_d)/2
            if distance_s_2 > v_sh**2/a_sh:
                s_3_time_2 = 2*v_sh/a_sh+(distance_s_2-2*v_sh**2/(2*a_sh))/v_sh
            else:
                s_3_time_2 = 2*np.sqrt(distance_s_2/a_sh)

            s_3_time = s_3_time_1 + s_3_time_2 + t_sh
            return s_3_time

        def s_9(n_l, v_t, u_w, t_t, a_t):

            distance_c = 0
            for i in range(1, n_l+1):
                for j in range(1, n_l+1):
                    distance_c += (abs(i-j)*u_w) / (n_l**2)
            if distance_c > v_t**2/a_t:
                time_c = 2*v_t/a_t+(distance_c-2*v_t**2/(2*a_t))/v_t
            else:
                time_c = 2 * np.sqrt(distance_c / a_t)
            s_9_time = time_c + 2*t_t
            return s_9_time

        def c_time(n_l, v_t, u_w, a_t):

            distance_c = (n_l * u_w) / 2
            if distance_c > v_t ** 2 / a_t:
                s_c_time = 2 * v_t / a_t + (distance_c - 2 * v_t ** 2 / (2 * a_t)) / v_t
            else:
                s_c_time = 2 * np.sqrt(distance_c / a_t)
            return s_c_time

        def sh_time(n_c, v_sh, u_d, a_sh):

            distance_s = ((n_c-1)*u_d)/2
            if distance_s > v_sh ** 2 / a_sh:
                sh_time = 2 * v_sh / a_sh + (distance_s - 2 * v_sh ** 2 / (2 * a_sh)) / v_sh
            else:
                sh_time = 2 * np.sqrt(distance_s / a_sh)
            return sh_time

        def s_c(n_l, v_t, u_w, a_t):

            def c_2(n_l, v_t, u_w, a_t):
                distance_c = 0
                for i in range(1, n_l+1):
                    for j in range(1, n_l+1):
                        distance_c += (abs(i - j) * u_w) / (n_l ** 2)
                if distance_c > v_t ** 2 / a_t:
                    c_2_time = 2 * v_t / a_t + (distance_c - 2 * v_t ** 2 / (2 * a_t)) / v_t
                else:
                    c_2_time = 2 * np.sqrt(distance_c / a_t)
                return c_2_time

            s_c_1 = c_time(n_l, v_t, u_w, a_t)
            s_c_2 = c_2(n_l, v_t, u_w, a_t)
            s_c_3 = 0
            s_c_4 = s_c_1
            s_c_5 = s_c_1
            s_c_6 = c_2(n_l, v_t, u_w, a_t)
            s_c_7 = s_c_1
            s_c_8 = c_2(n_l, v_t, u_w, a_t)
            s_c_9 = 0
            s_c_10 = s_c_1
            s_c_11 = s_c_1
            s_c_12 = c_2(n_l, v_t, u_w, a_t)

            s_c_vector = np.array([s_c_1,s_c_2,s_c_3,s_c_4,s_c_5,s_c_6,s_c_7,s_c_8,s_c_9,s_c_10,s_c_11,s_c_12])
            s_p_vector = np.array([self.p_1,self.p_2,self.p_3,self.p_4,self.p_5,self.p_6,self.p_7,self.p_8,self.p_9,self.p_10,self.p_11,self.p_12])
            s_c_time = s_c_vector.dot(s_p_vector)
            s_c_time_2 = (s_c_vector*s_c_vector).dot(s_p_vector)
            s_c_scv = s_c_time_2 / s_c_time**2 - 1
            return s_c_time, s_c_scv

        S_1 = sh_time(n_c, v_sh, u_d, a_sh)
        S_2 = t_sh
        S_3 = s_3(n_c, v_sh, u_d, t_sh, a_sh)
        S_4 = S_1 + t_sh
        S_0 = 0
        s_s_vector = np.array([S_1,S_1,S_2,S_2,S_3,S_3,S_1,S_1,S_0,S_0,S_4,S_4])
        s_p_vector = np.array([self.p_1,self.p_2,self.p_3,self.p_4,self.p_5,self.p_6,self.p_7,self.p_8,self.p_9,self.p_10,self.p_11,self.p_12])
        self.s_3_mean = (2 * n_l - 1) / (2*n_l) * S_3 + 1 / (2*n_l) * S_4
        self.s_3_dev = np.sqrt((2 * n_l - 1) / (2*n_l) * S_3**2 + 1 / (2*n_l) * S_4**2 - self.s_3_mean**2)

        self.s_5 = 2*c_time(n_l, v_t, u_w, a_t) + 2*t_t + t_sh
        self.s_6 = c_time(n_l, v_t, u_w, a_t) + 2*t_t
        self.s_7 = sh_time(n_c, v_sh, u_d, a_sh) + t_sh
        self.s_8 = t_sh
        self.s_9 = s_9(n_l, v_t, u_w, t_t, a_t)
        self.s_10 = c_time(n_l, v_t, u_w, a_t) + 2*t_t
        self.s_s = s_s_vector.dot(s_p_vector)
        self.s_c, self.s_c_scv = s_c(n_l, v_t, u_w, a_t)
        self.s_s_scv = (s_s_vector*s_s_vector).dot(s_p_vector)/self.s_s**2 - 1

        self.p_a = self.p_1 + self.p_2
        self.p_b = self.p_3 + self.p_4
        self.p_c = self.p_5 + self.p_6
        self.p_chain = self.p_7 + self.p_8 + self.p_9 + self.p_10 + self.p_11 + self.p_12
        self.s_2 = self.p_a*self.s_5 + self.p_b*self.s_6 + self.p_c*self.s_6 + self.p_chain*(0.25*self.s_9+0.25*self.s_10+0.5*self.s_6)
        s_2_2 = self.p_a*self.s_5**2 + self.p_b*self.s_6**2 + self.p_c*self.s_6**2 + self.p_chain*(0.25*self.s_9**2+0.25*self.s_10**2+0.5*self.s_6**2)
        self.s_2_scv = s_2_2 / self.s_2**2 - 1

    def kron_sum(self, x, y):

        # define kronecker sum

        if x.shape[0] == x.shape[1] and y.shape[0] == y.shape[1]:
            i_x = sp.identity(x.shape[0])
            i_y = sp.identity(y.shape[0])
            kron_sum_xy = sp.kron(x, i_y) + sp.kron(i_x, y)
            return kron_sum_xy

    def fork_join(self, m):

        # estimate performance of fork-join network

        k_s = int(np.ceil(1 / self.s_s_scv))                         # phase number of shuttles
        k_c = int(np.ceil(1 / self.s_c_scv))                         # phase number of the transfer car
        mu_s = k_s / self.s_s                                        # service rate of shuttles
        mu_c = k_c / self.s_c                                        # service rate of the transfer car
        t_s = np.diag([-mu_s]*k_s) + np.diag([mu_s]*(k_s-1), 1)
        t_c = np.diag([-mu_c]*k_c) + np.diag([mu_c]*(k_c-1), 1)
        i_s = np.identity(k_s)
        i_c = np.identity(k_c)
        t_s_0 = np.zeros(k_s).reshape(-1, 1)
        t_s_0[-1] = mu_s
        t_c_0 = np.zeros(k_c).reshape(-1, 1)
        t_c_0[-1] = mu_c
        alpha_s = np.zeros(k_s).reshape(1, -1)
        alpha_s[0, 0] = 1
        alpha_c = np.zeros(k_c).reshape(1, -1)
        alpha_c[0, 0] = 1
        t_c_0_1 = sp.kron(sp.block_diag(t_c_0), i_s)
        t_s_0_1 = sp.block_diag(t_s_0)

        # define the transition matrix

        b01_item = sp.kron(t_c_0, i_s)

        b10_item = sp.kron(alpha_c, t_s_0.dot(alpha_s))

        b00_0_item1 = self.kron_sum(t_c, t_s)
        b00_0_item2 = self.kron_sum(t_c, t_s) - t_c_0_1
        b00_0_item3 = t_c

        b00_1_item1 = sp.kron(i_c, t_s_0.dot(alpha_s))
        b00_1_item2 = sp.kron(i_c, t_s_0)

        b00_2_item1 = sp.kron(t_c_0.dot(alpha_c), i_s)
        b00_2_item2 = sp.kron(t_c_0.dot(alpha_c), alpha_s)

        b11_0_item1 = t_s - t_s_0_1
        b11_0_item2 = t_s

        b11_1_item1 = t_s_0.dot(alpha_s)

        def b_01_fun():

            if m == 1:
                b01 = b01_item
            else:
                b01 = b01_item
                for i in range(m-1):
                    b01 = sp.bmat([[b01, None], [None, b01_item]])
            b_01 = sp.vstack([b01, np.zeros([k_c, b01.shape[1]])])
            return b_01

        def b_10_fun():

            b10 = b10_item
            for i in range(m-1):
                b10 = sp.bmat([[b10, None], [None, b10_item]])
            b_10 = sp.hstack([b10, np.zeros([b10.shape[0], k_c])])
            return b_10

        def b_00_fun():

            def b00_diag_0():

                if m == 1:
                    b00_diag_0 = sp.bmat([[b00_0_item1, None], [None, b00_0_item3]])
                else:
                    b00_diag_0 = b00_0_item1
                    for i in range(m-1):
                        b00_diag_0 = sp.bmat([[b00_diag_0, None], [None, b00_0_item2]])
                    b00_diag_0 = sp.bmat([[b00_diag_0, None], [None, b00_0_item3]])
                return b00_diag_0

            def b00_diag_1():

                if m == 1:
                    b00_diag_1 = b00_1_item2
                else:
                    b00_diag_1 = b00_1_item1
                    for i in range(m-2):
                        b00_diag_1 = sp.bmat([[b00_diag_1, None], [None, b00_1_item1]])
                    b00_diag_1 = sp.bmat([[b00_diag_1, None], [None, b00_1_item2]])
                b00_diag_1 = sp.bmat([[None, b00_diag_1], [np.zeros([b00_0_item3.shape[0], b00_0_item1.shape[1]]), None]])
                return b00_diag_1

            def b00_diag_2():

                if m == 1:
                    b00_diag_2 = b00_2_item2
                else:
                    b00_diag_2 = b00_2_item1
                    for i in range(m-2):
                        b00_diag_2 = sp.bmat([[b00_diag_2, None], [None, b00_2_item1]])
                    b00_diag_2 = sp.bmat([[b00_diag_2, None], [None, b00_2_item2]])
                b00_diag_2 = sp.bmat([[None, np.zeros([b00_0_item1.shape[0], b00_0_item3.shape[1]])], [b00_diag_2, None]])
                return b00_diag_2

            b_00 = b00_diag_0() + b00_diag_1() + b00_diag_2()
            return b_00

        def b_11_fun():

            def b11_diag_0():

                if m == 1:
                    b11_diag_0 = b11_0_item2
                else:
                    b11_diag_0 = b11_0_item1
                    for i in range(m-2):
                        b11_diag_0 = sp.bmat([[b11_diag_0, None], [None, b11_0_item1]])
                    b11_diag_0 = sp.bmat([[b11_diag_0, None], [None, b11_0_item2]])
                return b11_diag_0

            def b11_diag_1():

                b11_diag_1 = b11_1_item1
                if m > 1:
                    for i in range(m - 2):
                        b11_diag_1 = sp.bmat([[b11_diag_1, None], [None, b11_1_item1]])
                    b11_diag_1 = sp.bmat([[None, b11_diag_1], [np.zeros([b11_0_item2.shape[0], b11_0_item1.shape[1]]), None]])
                return b11_diag_1

            if m == 1:
                b_11 = b11_diag_0()
            else:
                b_11 = b11_diag_0() + b11_diag_1()
            return b_11

        b_00 = b_00_fun()
        b_01 = b_01_fun()
        b_10 = b_10_fun()
        b_11 = b_11_fun()

        pai_c_0 = b_00.shape[0]

        q = sp.bmat([[b_00, b_01], [b_10, b_11]])
        q_1 = sp.hstack([q, np.mat([[1]]*q.shape[0])])
        b = np.zeros(q_1.shape[1]).reshape(-1, 1)
        b[-1] = 1
        pai = li.lsqr(q_1.T, b)[0]

        fk_tp = 0
        pai0 = pai[:pai_c_0]
        pai1 = pai[pai_c_0:]
        for i in range(1, 2*m+1):
            if i < m:
                fk_tp += pai0[i*k_c*k_s:(i+1)*k_c*k_s].sum() * (mu_c / k_c) * (i/m)
            elif i == m:
                fk_tp += pai0[-k_c:].sum() * (mu_c / k_c)
            elif m < i < (2 * m):
                fk_tp += pai1[(i - m - 1) * k_s:(i - m) * k_s].sum() * (mu_s / k_s) * (1 / (2*m-i+1))
            else:
                fk_tp += pai1[-k_s:].sum() * (mu_s / k_s)
        return fk_tp

    def mva(self, m):

        # estimate the performance of closed network

        s_t = np.mat([self.s_2, (self.p_a+self.p_b)*self.s_7, (self.p_c+self.p_chain)*self.s_8])

        visit_ratio = np.mat([1 + self.p_chain, 1 + self.p_chain, 1, 1])

        def tp_k():

            la_k = 0
            wt_fj = 0
            wt_2 = 0
            pai_0 = np.mat([[0]*(m+1)]*(m+1), dtype='float64')
            pai_0[0, 0] = 1
            mu_fj_k = np.mat([[0]]*(m+1), dtype='float64')
            k_2 = np.mat([0]*(m+1), dtype='float64')
            for i in range(m):
                mu_fj_k[i] = (i+1) / self.fork_join(i+1)
            for k in range(m):
                wt_fj = float(pai_0[k].dot(mu_fj_k))
                wt_2 = s_t[0,0]*(1+k_2[0, k])
                m_s_t = np.dot(np.mat([wt_fj, wt_2, s_t[0,1], s_t[0,2]]), visit_ratio.T)
                la_k = (k+1) / float(m_s_t)
                k_2[0, k+1] = la_k * wt_2 * visit_ratio[0,1]
                for j in range(k+1):
                    pai_0[k+1, j+1] = (la_k * pai_0[k, j] * mu_fj_k[j+1] * visit_ratio[0,0]) / (j+1)
                pai_0[k+1, 0] = 1 - pai_0[k+1].sum()
            return la_k, wt_fj, wt_2

        tp_mva, wt_fj, wt_2 = tp_k()

        # estimate the mean queue length

        m_q_l = float(tp_mva) * np.multiply(visit_ratio, np.mat([wt_fj, wt_2, s_t[0,1], s_t[0,2]]))

        return tp_mva, m_q_l

    def sys_tp(self):

        # estimate the system performance

        mu_n = []
        ql_in = np.zeros(self.n_s, dtype='float64')
        for i in range(1, self.n_s + 1):
            a, b = self.mva(i)
            mu_n.append(self.la / a)
            ql_in[i-1] = b[0, [0,1]].sum()
        mu_n_acumulate_mul = np.array(list(it.accumulate(mu_n, lambda x, y: x*y)))

        pai_n = 1 / (1 + mu_n_acumulate_mul[:-1].sum() + mu_n_acumulate_mul[-1] * ((self.la / mu_n[-1]) / ((self.la / mu_n[-1]) - self.la)))

        pai_n_0 = np.zeros(self.n_s, dtype='float64')
        for i in range(self.n_s):
            pai_n_0[i] = pai_n * mu_n_acumulate_mul[i]
        pai_0 = np.append(pai_n, pai_n_0)[::-1]

        m_q_l_n = np.arange(self.n_s+1).dot(pai_0)                         # mean queue length of idle shuttles
        m_q_l_j = pai_0[0] * (mu_n[-1] / (1 - mu_n[-1])**2)                # external queue length

        u_c = 0
        j = 0
        for i in ql_in[::-1]:
            if i == 0:
               u_c += pai_n_0[j]
            j += 1
        u_c = 1 - pai_n - u_c                                              # utilization of the transfer car

        u_s = 1 - m_q_l_n / self.n_s                                       # utilization of shuttles
        e_tp = m_q_l_j / self.la + (self.n_s - m_q_l_n) / self.la          # system throughput time
        return e_tp
