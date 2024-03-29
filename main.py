import numpy as np

class Simplex:
    def __init__(self, vertices):
        self.vertices = vertices
        self.size = len(vertices)

    def __lt__(self, other):
        if self.size == other.size:
            for i in range(0, self.size):
                if self.vertices[i] < other.vertices[i]:
                    return True
                elif self.vertices[i] > other.vertices[i]:
                    return False
            return False
        else:
            return self.size < other.size


# Abstract Simplicial Complex
class ASC:

    def __init__(self, simplices):
        length = max([len(a) for a in simplices])
        simplices_array = [[] for _ in range(0, length)]
        for s in simplices:
            s.sort()
            S = Simplex(s)
            simplices_array[S.size - 1].append(S)
        for i in simplices_array:
            i.sort()
        self.simplices = simplices_array

    def simplex_boundary(self, S):
        if S.size <= 1:
            return []
        dim = S.size - 1

        k = len(self.simplices[dim-1])
        rv = np.zeros(k)
        k -= 1

        for (j, v) in enumerate(S.vertices):
            face = [k for k in S.vertices if k!=v]

            while self.simplices[dim-1][k].vertices != face:
                k -= 1

            rv[k] = (-1)**j

        return rv

    
    # gives the boundary operator in terms of canonical bases
    def simplex_matrix(self, i:int):
        if i == 0:
            return np.zeros((1, len(self.simplices[i])))

        if i < 0:
            return np.empty((1,1))

        #m x n, m the number of generators of C_(i-1), n the number of C_i
        M = np.zeros((len(self.simplices[i-1]), len(self.simplices[i])))
        for (j, S) in enumerate(self.simplices[i]):
            j_bdy = self.simplex_boundary(S)
            M[:,j] = j_bdy

        return M

    def H_n(self, n):
        M_n = self.simplex_matrix(n)

        (_, D_n, Q_n) = smith_normal_form(M_n)
        ker_dim = M_n.shape[1] - len(D_n[D_n!=0])

        if ker_dim == 0:
            return []
        #Last ker_dim columns of Q form basis for kernel
        ker_basis = Q_n[:, -ker_dim:]

        #If we're in the top homology group, return the kernel
        if n+1 > len(self.simplices) - 1:
            pfs = []
            for b in ker_basis.T:
                ss = []
                for (k, c) in enumerate(b):
                    if c != 0:
                        ss.append((c, self.simplices[n][k]))
                pfs.append((0, ss))
            return pfs
        
        M_n_1 = self.simplex_matrix(n+1)
        (P, D, _) = smith_normal_form(M_n_1)


        tor = np.abs(D[D!=0])
        im_dim = len(tor)

        #first im_dim columns of P^-1 form a basis for the image
        im_basis = np.linalg.inv(P)[:, :im_dim]

        pfs = []
        for (i, a) in enumerate(tor):
            #Decompose composites with Chinese remainder theorem if the torsion isn't trivial
            if a != 1:
                pfs_ = pf(a)
                for (p, j) in pfs_:
                    pj = p**j
                    #turn im_basis into list of simplices
                    b = (a // pj)*im_basis[:, i]
                    ss = []
                    for (k, c) in enumerate(b):
                        if c != 0:
                            ss.append((c, self.simplices[n][k]))
                    pfs.append((pj, ss))

        #pfs.sort(key=lambda x: x[0])
        if ker_dim != im_dim:
            free_basis = basis_completion(im_basis, ker_basis)
            for v in free_basis:
                pfs.append((0, v))

        return pfs


def smith_normal_form(M):
    (m,n) = M.shape
    A = np.identity((m+n), dtype='int64')
    A[0:m, m:m+n] = M

    for l in range(0, min(m,n)):

        (min_i, min_j) = min_SNF(A[l:m, m+l:m+n])
        min_i += l # type: ignore
        min_j += l + m

        #swap rows+columns to put min at (l, m+l) in A
        A[[l, min_i], :] = A[[min_i, l], :]
        A[:, [m+l, min_j]] = A[:, [min_j, m+l]]

        # check if column/row is zeroed out
        while np.any(A[l+1:m, m+l]) or np.any(A[l,m+l+1:]):

            # zero out column
            for i in range(l+1, m):
                if A[i, m+l] % A[l, m+l] != 0:
                    (a,b,_) = gcd(A[l, m+l], A[i, m+l])
                    (c, d, _) = gcd(a, b)

                    E = np.eye(m+n, dtype='int64')
                    E[l, l] = a
                    E[l, i] = b
                    E[i, i] = c
                    E[i, l] = -d



                    A = E@A
                
            for i in range(l+1, m):
                q = A[i, m+l] // A[l, m+l]
                A[i, :] -= q*A[l, :]

            #zero out row
            for j in range(m+l+1, m+n):
                if A[l, j] % A[l, m+l] != 0:
                    (a,b, _) = gcd(A[l, m+l], A[l, j])
                    (c, d, _) = gcd(a, b)

                    E = np.eye(m+n, dtype='int64')
                    E[l, l] = a
                    E[l, j] = -d
                    E[j, j] = c
                    E[j, l] = b

                    A = A@E
                
            for j in range(m+l+1, m+n):
                q = A[l, j] // A[l, m+l]
                A[:, j] -= q*A[:, m+l]

    #arrange diagonal to order by divisibility
    #reduce_diagonal(A, m, n)

    P = A[:m, :m]
    D = A[:m, m:]
    Q = A[m:, m:]

    return (P, D, Q)


#INCOMPLETE Needed for SNF but can use non-divisible coefficients to calculate H_n
def reduce_diagonal(A, m, n):
    #TODO need to compare every two entries
    #Maybe a smarter way of doing this
    for l in range(0, min(m,n) - 1):
        if A[l+1, m+l+1] != 0 and A[l+1, m+l+1] % A[l, m+l] != 0:
            (a, b, _) = gcd(A[l, m+l], A[l+1, m+l+1])
            (c, d, _) = gcd(a,b)

            A[:, m+l] += A[:, m+l+1]

            E = np.eye(m+n, dtype='int64')
            E[l, l] = a
            E[l, l+1] = b
            E[l+1, l+1] = c
            E[l+1, l] = -d

            A = E@A
            A[l+1, :] -= (A[l+1, m+l]) // (A[l, m+l]) * A[l, :]
            A[:, m+l+1] -= (A[l, m+l+1]) // (A[l, m+l]) * A[:, m+l]



def min_SNF(M):
    mM = np.abs(np.ma.masked_array(M, mask=M==0))
    rv = np.unravel_index(np.argmin(mM, axis=None), mM.shape)
    return (rv[0], rv[1])

#returns (a, b, g) where g = gcd(m, n) and am+bn = g
#implements euclidean algorithm
def gcd(m, n):
    g_1 = m
    a_1 = 1
    b_1 = 0
    g_2 = n
    a_2 = 0
    b_2 = 1

    while g_2 != 0:
        g_2_tmp = g_2
        a_2_tmp = a_2
        b_2_tmp = b_2

        q = g_1 // g_2

        g_2 =g_1 - q*g_2_tmp
        a_2 = a_1 - q*a_2_tmp
        b_2 = b_1 - q*b_2_tmp

        g_1 = g_2_tmp
        a_1 = a_2_tmp
        b_1 = b_2_tmp

    if g_1 < 0:
        return (-a_1, -b_1, -g_1)
    else:
        return (a_1, b_1, g_1)

def pf(m):
    rv = []
    a = 2
    while m > 1:
        if a**2 > m:
            rv.append((m, 1))
            break

        i = 0
        while m % a == 0:
            i += 1
            m //= a

        if i > 0:
            rv.append((a, i))

        a += 1

    return rv


#Given a space B generated by bs lying in an ambient free Z-module, extend vs to generate bs
#B an nxk matrix, n rank of free z module and k rank of B, columns are basis
#vs an nxl matrix, n rank of free Z-module and l rank of space generated by vs, columns are basis
def basis_completion(vs, B):
    num_vs = vs.shape[1]
    #Step 1: write vs in terms of bs
    B_ = np.linalg.pinv(B)
    v_bs = (np.rint(np.matmul(B_, vs))).astype(int)


    #Step 2: extract basis completion from SNF in terms of bs
    (P, D, Q) = smith_normal_form(v_bs)
    v_completion = np.linalg.inv(P)[:, num_vs:]

    #Step 3: write the completion in terms of standard coordinates
    rv = B@v_completion
    return rv

#M = np.array([[10,15,4,3,0,8],[5,31,7,12,3,4],[42,16,0,1,1,8],[0,5,18,17,30,48]])
#N = np.array([[2,4,4],[-6,6,12],[10,4,16]])
#smith_normal_form(N)

#S_1 = [[0,1],[0,2],[1,2],[0],[1],[2]]
#S_2 = [[0,1,2],[0,1,3],[0,2,3],[1,2,3],[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0],[1],[2],[3]]
#RP_2 = [[0,1,4], [0,1,5], [0,2,3], [0,2,4], [0,3,5], [1,2,3], [1,2,5], [1,3,4], [2,4,5], [3,4,5], [0,1], [0,2], [0,3], [0,4], [0,5], [1,2], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [3,5], [4,5], [0], [1], [2], [3], [4], [5]]

#S = ASC(RP_2)
#S.H_n(1)