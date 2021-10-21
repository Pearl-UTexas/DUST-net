############################################################################################
#  Part I:                          Calculation zonal C polynomials                        #
#  By                               Lin JIU                                                #
#  Project Initiate:                March 1st, 2017                                        #
#  Last Update:                     January 31st, 2020                                     #
############################################################################################

############################################################################################
##    Section I: Some general functions that can be used later for other packages         ##
############################################################################################

def twolistsfunction(l1, l2, fun):
    '''
    Given two lists l1 and l2 of the same length and a function fun with two variables, return the list of result by applying fun component-wise on l1 and l2.
    
    For example, if l1 = [1, 2, 3], l2 = [4, 5, 6] and fun = operator.add, then the function returns [1 + 4, 2 + 5, 3 + 6] = [5, 7, 9].

    Remark: 1) The map_threaded function only apply function with one variable to a list. 
            2) Use help(operator) to find more operators. 
    '''
    
    n1 = len(l1)                                              # The length of l1#
    n2 = len(l2)                                              # The lenght of l2#
    if n1 != n2:
        return "error"                                        # If two lists have different lengths, return error -1#
    else:
        return [fun(l1[t], l2[t]) for t in range(n1)]         # Otherwise, return the desired list#

def APPENDZERO(l1, l2):
    '''
    Given two lists with possible different length, add zero(s) to the shorter one, to make them have the same length. 

    For example, l1 = [1] and l2 = [2, 3, 4]. Then, 
    APPENDZERO(l1, l2) = [[1, 0, 0], [2, 3, 4]].
    '''
    
    n1 = len(l1)    # The length of l1 #
    n2 = len(l2)    # The length of l2 #
    if n1 == n2:
        return [l1, l2]
    elif n1 > n2:
        return [l1, l2 + [0 for i in range(n1-n2)]]
    else:
        return [l1 + [0 for i in range(n2-n1)], l2]


############################################################################################
##    Section II: M(ominal) symmetric polynomials                                         ##
############################################################################################



def Calmi(p):                        
    '''
    Give a paritition p of n, with m_1 1's, m_2 2's and etc, return (m_1)!(m_2)!...(m_n)!. 
    This is the reciprocal of the leading coefficient of M-polynomial.
    
    For example, if  p = [5, 4, 4, 3, 1] = (1^1, 2^0, 3^1, 4^2, 5^1), Calmi(p) = 1! * 0! * 1! * 2! * 1! = 2. 
    '''
    
    t = Partition(p).to_exp()               # exponential form of the partition (1^{m_1}, 2^{m_2},..., n^{m_n}) #
    return prod(factorial(i) for i in t)    # return the product (m_1)!(m_2)!...(m_l)! #


def MZonal(partition, variables):              
    '''
    Given a partition p and list of variables l, return the M-polynomial M_p(l).

    For example, if p = [2, 1] and l = [a, b, c, d], by defintion, 
    M_p(a, b, c, d) = a^2b + a^2c + a^2d + b^2c + b^2d + c^2d + symmetric terms
                    = a^2*b + a*b^2 + a^2*c + b^2*c + a*c^2 + b*c^2 + a^2*d + b^2*d + c^2*d + a*d^2 + b*d^2 + c*d^2

    '''

    ind = 0;                                                              #index is 0 or 1. #
    re = 0;                                                               #return#
    m = len(partition);                                                   #m=length of partition#
    n = len(variables);                                                   #n=number of variables#
    if n < m:                                                             #If n<m, index is 1, there are more parts in the partition than variables, hence return 0#
        ind = 1;
    if ind == 0:                                                          #If index is 0, not 1, continue to compute.#
        perm = Permutations(n,m).list();                                  #For the summation part, we choose all posible combination of the variables for the summand#
        for i in perm:                                                    #For each summmand,#
            temp = [variables[j-1] for j in i];                           #pick the right combination of the variables#
            re += prod(twolistsfunction(temp, partition, operator.pow))   #Get the term (y_1)^(\lambda_1)(y_2)^(\lambda_2)...(y_m)^(\lambda_m)#
        re = re / Calmi(partition)                                        #Divide the sum by the leading coefficient.#                      
    return re;

############################################################################################
##    Section III: Calculation of C-Zonal polynomials                                     ##
############################################################################################

## Since all the C-polynomials are linear combinations of M-polynomials, as long as the coefficients are computed, the polynomial is easy to obtain. ##


def RHO(p):
    '''
    Given a partition p = (p_1, p_2, ..., p_m), return the value of rho_p = sum_{i=1}^m p_i * (p_i -i).
    Note that position in Sage begins from 0, explaining the "-i-1" instead of "-i" in the code. 

    For example, p = [4, 2, 2], then rho_p = 4 * (4-1) + 2 * (2-2) + 2 * (2-3) = 10.
    '''

    return sum(p[i] * (p[i] - i - 1) for i in range(len(p)));


def CHI(p, n):
    '''
    Given a partition p = (p_1, p_2, ..., p_m) of n, return the value of chi_{[2p]}(1), defined by the following ratio:

    (2n)! * prod_{i,j = 1, i < j}^m (2p_i - 2p_j - i + j)
    -------------------------------------------------------
    prod_{i = 1}^m (2p_i + m -i)!

    This is one of the key factor to compute c_{p,p} in the function COE. 
    '''

    m = len(p)      #The number of parts#
    re = factorial(2 * n) * prod(prod(2 * p[i - 1] - 2 * p[j - 1] - i + j for i in range(1, j)) for j in range(2, m + 1))/prod(factorial(2 * p[i] + m - (i + 1)) for i in range(m))
    return re


def COE(k, n):
    '''
    Given a partition k of integer n, return the coefficient c_{k,k}. It is easy if k = (n), with c_{(n), (n)} = 1.
    '''

    m = len(k)     # The number of parts#
    if m == 1: 
        re = 1     # If k has only one part, i.e., k = (n), then c_{(n), (n)} = 1
    else:
        t = k + [0]     # Add a 0 in the end, due to the formula. t is short for temp#
        re = 2^(2 * n) * factorial(n)/factorial(2 * n) * CHI(k, n) * prod(prod(rising_factorial(l/2 - (i - 1)/2 + t[i - 1] - t[l - 1], t[l - 1] - t[l]) for i in range(1, l + 1)) for l in range(1, m + 1))
#   As one can see that in the formula, there is a index k[l+1], which could exceed the length of k. Therefore, use t = k + [0] instead #
    return re


#To use the recurrence of the coefficients, we first compute the set of mu that will appear in the summation.#

def SumVariable(k, l):                             
    '''
    Given two partitions k and l = (lambda_1, ..., lambda_m) of n, it returns for all possible partition mu that appears in the recurrence for c_{k, l}.
    For each mu, it returns the pair (list) [lamda_i-lambda_j+2t, mu].


    For example, if k = [4, 2, 2] and l = [3, 3, 1, 1]. 
    SumVariable(k, l, n) = [[2, [4, 2, 1, 1]], [2, [3, 3, 2]]]
    
    Another tricky example is k = [4, 3, 1] and l = [3, 3, 1, 1]. 
    The output comtains [4, 3, 1] FOUR times, since it can be obtain in 4 different ways from l. 
    '''

    m = len(l)                                                    # m = length of partition l#
    re = []
    for i in range(m - 1): 
        for j in range(i + 1, m):                                 # (i, j) runs over all pars with i<j, correspondng to lambda_i and lambda_j of \mu#
            for t in range(1, l[j] + 1):                          # t = 1, 2, ..., lambda_j #
                tem = [a for a in l];
                tem[i] = tem[i] + t;
                tem[j] = tem[j] - t;
                tem = sorted(tem, reverse = true);                # Reorder temp #
                tem = [temp2 for temp2 in tem if temp2 != 0];     # remove all zeros #
                if tem > l and tem <= k:
                    re = re + [[l[i] - l[j] + 2 * t, tem]];
    return re;



def Lcoeffi(listKtoL):                      
    '''
    The input is a list of partitions of integer n, which begins with partition k and ends with partition l, such that k > l. 
    Also, the list contains all the partitions mu, with k >= mu >= l, in the lexicographic order. 
    The output is a sequence of coefficients c_{k, mu} for mu in the input list. 

    For example, let k = [4, 1] and l = [2, 1, 1, 1]. Then
    1) the input should be [[4, 1], [3, 2], [3, 1, 1], [2, 2, 1], [2, 1, 1, 1]];
    2) the output is []
    '''
    
    m = len(listKtoL);                   # length of this partial list#
    re = [1/2 for x in range(m)];        # list of c_{k,mu} for all mu between k and l. Initially set 1/2, since coefficients are rational numbers #
    k = listKtoL[0];                     # the first partition k#
    re[0] = COE(k, sum(k));              # c_{k,k}#
    for x in range(1,m):                 # start from mu < k#
        mu = listKtoL[x]; 
        rho = RHO(k) - RHO(mu);
        if rho <= 0:                     # Due to the positivity, and Remark 1 of Takemura's Chapter 4, on P. 73, if rho <= 0, the coefficient is set to be 0 #
            re[x] = 0;
        else:
            table = SumVariable(k, mu);  # Find the table of #
            y = len(table);
            temp1 = [1/2 for t in range(y)];
            for t in range(y):
                temp1[t] = listKtoL.index(table[t][1]);                    # find the position of the mu in the list, in order to apply the recurrence#
            re[x] = sum(table[t][0] * re[temp1[t]] for t in range(y))/rho;
    return re;   

def Coeffi(k, l): 
    '''
    Given two partitions k and l of integer n, compute the coefficient c_{k, l}.

    
    For example, k = [4, 1] and l = [2, 2, 1]. Then, Coeffi(k, l) = 4. 
    '''

    n = sum(k);                       # sum of the partition #
    if l == k:
        return COE(k, n);
    elif l > k:
        return 0;
    else:
        L = Partitions(n).list()               # the full list of all partitions of n #
        p1 = L.index(k)                        # position of k #
        p2 = L.index(l)                        # position of l #
        L = [list(x) for x in L[p1: p2 + 1]]   # the list of partitions from k to l, including both ends #
        return Lcoeffi(L)[-1];                 # The last one is c_{k, l} #

# Since the coefficients and M-polynomials are ready, we can now compute the C-polynomials #

def CZonal(k,v):    
    '''
    Given partition k and variables v, compute C-polynomial C_{k}(v).

    For example, 
    CZonal([4, 1, 1], [a, b, c]) = 16*a^4*b*c + 48/5*a^3*b^2*c + 48/5*a^2*b^3*c + 16*a*b^4*c + 48/5*a^3*b*c^2 + 32/5*a^2*b^2*c^2 + 48/5*a*b^3*c^2 
                                   + 48/5*a^2*b*c^3 + 48/5*a*b^2*c^3 + 16*a*b*c^4
    '''
    
    n = sum(k);                                                             # k is a partition of n $
    whole = Partitions(n).list();                                           # whole list of partitions of n #
    p = whole.index(k);                                                     # position of k in the whole list #
    partiallist = whole[p:];                                                # only consider partitions <= k #
    coefftable = Lcoeffi(partiallist);                                      # list of all coefficients c_{k,l} for l <=  k#
    Mtable = [MZonal(list(t),v) for t in partiallist];                      # list of all corresponding M_{l}(v )#
    re = sum(coefftable[t]  * Mtable[t] for t in range(len(partiallist)));  # linear combination of M-polynomials #
    return re;


##################################################################################
# Part II:                         Further Extensions to J, P and Q polynomials  #
# By:                              Primary: Raymond Kan, Secondary: Lin Jiu      #
# Begin Date:                      November, 2018                                #
# Last Update:                     January, 2019                                 #
##################################################################################



#Given a partition p=[a_1,a_2,a_3,...,a_l] of n, first convert it into the form of [m_1,m_2,...,m_n], where m_k stands for the number of k's appearing in p, which can be 0. #
#Then, return the product (m_1)!(m_2)!...(m_n)!. This is the reciprocal of the leading coefficient of M-polynomial.#
#def Calmi(p):                         
#    t=Partition(p).to_exp();           #exponential form of the partition p, i.e., [a_1,a_2,a_3,...,a_l]->[m_1,m_2,...,m_n]#
#    re=prod(factorial(i) for i in t);  #re=(m_1)!(m_2)!...(m_l)!#
#    return re;                         #return the product#

def listexp(list1,list2):                       #Given two lists of same length, say {a_1,a_2,...,a_n} and {b_1,b_2,...,b_n} return (a_1)^(b_1)(a_2)^(b_2)...(a_n)^(b_n)#
    n=len(list1);       
    re=[list1[i]^list2[i] for i in range(n)];
    return prod(i for i in re); 

# A function to find out if partition kappa dominates or equal to partition lam#
def dominate(kappa,lam):
     if len(kappa)>len(lam):
         return False
     else:
         s1=0;                         #Partial sum of parts of partition kappa#
         s2=0;                         #Partial sum of parts of partition lam#
         for i in range(len(kappa)):
             s1=s1+kappa[i];
             s2=s2+lam[i];
             if s1<s2:
                 return False
         return True

# A function to find out if mu and lam have at most two distinct elements. It requires mu>=lam.
def checkmu(lam,mu):
    lam1=list(lam);
    m1=len(lam1);
    m2=len(mu);
    if (m2<m1-1)|(m2>m1):
        return False
    for i in range(m2):
        try:
            lam1.remove(mu[i])
        except:
            pass
    return len(lam1)<=2

def MZONAL(partition,variables):              #Computing M-polynomial by (3.2)#
    re=0;                                     #return#
    m=len(partition);                         #m=length of partition#
    n=len(variables);                         #n=number of variables#
    if n>=m:                                  #If n<m, index is 1 and return 0#
        perm=Permutations(n,m).list();        #By (3.2), for the summation part, we choose all posible combination of the variables for the summand#
        for i in perm:                        #For each summmand,#
            temp=[variables[j-1] for j in i]; #pick the right combination of the variables#
            re=re+listexp(temp,partition);    #Get the term (y_1)^(\lambda_1)(y_2)^(\lambda_2)...(y_m)^(\lambda_m)#
        re=re/Calmi(partition);                   #Divide the sum by the leading coefficient.#
    return re;

#Given partitions lam<kappa, return all nonzero c_{mu,lam} with kappa>=mu>=lam#
def LCoeffi(kappa,lam,*arg):                  
    if len(arg)==0:
        norm='J';
    else:
        norm=arg[0];     
    A=kappa.dominated_partitions();  #only consider partitions <=k#
    k=sum(kappa);
    if len(lam)<k:
        count=0;
        while A[count]>=lam:
            count=count+1;
        A=A[:count];    
    m=len(A);                #length of this partial list#
    M=list(zero_vector(m));
    M[0]=kappa.hook_product(2);
    if len(lam)==k:
        M[-1]=factorial(k);
    for i in range(m):
        A[i]=list(A[i]);
    kappa_l=[len(A[i]) for i in range(m)];
    rho=[sum([A[i][j]*(A[i][j]-j-1) for j in range(kappa_l[i])]) for i in range(m)];     
# Create a matrix with coefficients a_lam(mu)
    m1=m-ZZ(len(lam)==k);
# Compute the coefficients of M using recursion. 
    for jj in range(1,m1):
        lam1=A[jj];
        if dominate(kappa,lam1):
            m2=kappa_l[jj];
            c=0;
            for kk in range(jj):
                mu=A[kk];
                if checkmu(lam1,mu):
                    s=m2-1;
                    if kappa_l[kk]==m2:
                        while lam1[s]==mu[s]:
                            s=s-1;
                        t=mu[s];
                    else:
                        t=0;
                    while lam1[s]==mu[s-1]:
                        s=s-1;
                    t=lam1[s]-t;
                    r=s-1;
                    while (mu[r]!=lam1[r]+t)&((lam1[r]==mu[r])|(lam1[r-1]!=mu[r])):
                        r=r-1;
                    if lam1[r]==lam1[s]:
                        w=lam1.count(lam1[r]);
                        w=w*(w-1)/2;
                    else:
                        w=lam1.count(lam1[r])*lam1.count(lam1[s]);
                    c = c+(lam1[r]-lam1[s]+2*t)*w*M[kk];
            M[jj]=c/(rho[0]-rho[jj]); 
    if norm!='J':    
        if norm=='P':
            M=[M[i]/M[0] for i in range(m)];
        elif norm=='Q':
            c0=prod(flatten(kappa.upper_hook_lengths(2)));
            M=[M[i]/c0 for i in range(m)];
        else:
            c0=2**k*factorial(k)/prod(flatten(kappa.upper_hook_lengths(2)))/M[0];
            M=[M[i]*c0 for i in range(m)];
    return M

def COEFFI(k,l,*arg): 
    if len(arg)==0:
        norm='J';
    else:
        norm=arg[0];     
    if l>k:
        return 0;
    else:
        return LCoeffi(k,l,norm)[-1];    

def FLCoeffi(kappa,*arg):                      #Full List of coefficients c_{kappa,lambda} for all lambda<=kappa#
    if len(arg)==0:
        norm='J';
    else:
        norm=arg[0];
    k = sum(kappa);
    return LCoeffi(kappa,Partition([1]*k),norm)

def CZONAL(k,v,*arg):                                                     #Given partition k and variables v, compute C-polynomial C_{k}(v)#
    if len(arg)==0:
        norm='J';
    else:
        norm=arg[0];     
    partiallist=k.dominated_partitions();                                 #only consider partitions <=k#
    coefftable=FLCoeffi(k,norm);                                          #list of all coefficients c_{k,l} for l<=k#
    Mtable=[MZONAL(list(t),v) for t in partiallist];                      #list of all corresponding M_{l}(v)#
    re=sum(coefftable[t]*Mtable[t] for t in range(len(partiallist)));     #(3.3)#
    return re; 


# A function to compute the transition matrix from zonal polynomials to 
# monomials of degree k.  It takes two optional arguments:
# alpha: default i2 
# normalization: 'J','C','P', or 'Q', default is 'J'
def ZonaltoM(k,*arg):
    if len(arg)==0:
        norm='J';
    else:
        norm=arg[0];     
    A=Partitions(k).list();
    n=len(A);
    if (norm=='C')|(norm=='Q'):
        ck=[prod(flatten(A[i].upper_hook_lengths(2))) for i in range(n)]; 
# M is a transition matrix from J-normalization of zonal polynomials to monomials.
# It has the attractive property that all elements of this transition matrix are integers.
    M=identity_matrix(ZZ,n);
    for i in range(n):
        M[i,i]=A[i].hook_product(2);    # Use internal functions to compute the diagonal elements of M
        A[i]=list(A[i]);
    kappa_l=[len(A[i]) for i in range(n)];
    rho=[sum([A[i][j]*(A[i][j]-j-1) for j in range(kappa_l[i])]) for i in range(n)];     
    M[:,-1]=factorial(k);              # last column of M is k!
# Create a matrix with coefficients a_lam(mu)
    a=matrix(ZZ,n-1);
    for jj in range(1,n-1):
        lam=A[jj];
        m2=kappa_l[jj];
        for kk in range(jj):
            mu=A[kk];
            if checkmu(lam,mu):
                s=m2-1;
                if kappa_l[kk]==m2:
                    while lam[s]==mu[s]:
                        s=s-1;
                    t=mu[s];
                else:
                    t=0;
                while lam[s]==mu[s-1]:
                   s=s-1;
                t=lam[s]-t;
                r=s-1;
                while (mu[r]!=lam[r]+t)&((lam[r]==mu[r])|(lam[r-1]!=mu[r])):
                    r=r-1;
                if lam[r]==lam[s]:
                    w=lam.count(lam[r]);
                    w=w*(w-1)/2;
                else:
                    w=lam.count(lam[r])*lam.count(lam[s]);
                a[jj,kk]=(lam[r]-lam[s]+2*t)*w;
# Compute the coefficients of M using recursion. 
    for ii in range(n-1):
        kappa=A[ii];
        for jj in range(ii+1,n-1):
            lam=A[jj];
            if dominate(kappa,lam):
                M[ii,jj]=sum([a[jj,kk]*M[ii,kk] for kk in range(ii,jj)])/(rho[ii]-rho[jj]); 
    if norm!='J':    
        M=M.base_extend(QQ);
        if norm=='P':
            for ii in range(n):
                M[ii,ii:]=M[ii,ii:]/M[ii,ii];
        elif norm=='Q':
            for ii in range(n):
                M[ii,ii:]=M[ii,ii:]/ck[ii];
        else:
            c0=2**k*factorial(k);
            for ii in range(n):
                M[ii,ii:]=c0/M[ii,ii]/ck[ii]*M[ii,ii:]; 
    return M
