# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:25:23 2019

@author: Khai Xi
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy import stats

def uniformdist(n): #function to sample from a uniform distribution
    a = np.random.rand(n) #sample n number of samples from a uniform random distribution
    return a

def analytical(a): #function to run the analytical algorithm method
    b = np.arccos(2*a - 1) #returns a generated value from the normalised inverse sin function
    return b


def acceptroutine(k): #function to run the accept-reject method over an array of k number of points
    A = np.zeros(k) #create empty array of size k
    for i in range(k): #loop through number of points
        A[i] = acceptreject(1, 0) #runs the accept-reject method for each node
    return A

def acceptreject(mark, Zinput):
    """algorithm to run the accept-reject method for the entire task. Mark indicates which algorithm will be used for different parts of the question. Mark 1 is for Task 1, mark 2, is to generate z positions in Task 2, mark 3 is to generate x and y positions for Task 2, and mark 4 is the same as mark 3 except mark 4 has an added gaussian uncertainty. Zinput is only relevant for functions that generate x and y values."""
    if mark == 1:
        while True: #loop until condition is met
            B = np.random.uniform(0, np.pi) #sample unifrom distribution from 0 to pi
            C = np.random.rand() #sample uniform distribution
            D = np.sin(B) #run sin functions
            if C < D: #accept condition 
                return B
        
    if mark == 2:
        while True:  #loop until condition is met
            tau = 550*10**-6 #constant
            v = 2000 #constanr
            z = 2*np.random.rand() #sample z position from uniform distribution
            N = np.exp((-1/tau)*(z/v))/(v*tau) #normalised probabilistic function for radioactive decay
            B = np.random.rand() #random sample
            if B < N: #accept condition
                return z
      
    if mark == 3:
        while True: #loop until condition is met
            theta = np.random.uniform(0, 2*np.pi) # generating angle theta value
            phi = np.random.uniform(0, np.pi) #generating angle phi value
            x = (2-Zinput)*np.tan(theta)*np.cos(phi) #generating random x values
            y = (2-Zinput)*np.tan(theta)*np.sin(phi) #generaing random y values
            if abs(x) < 0.5 and abs(y) < 0.5: #accept condition
                return x, y
        
    if mark == 4:
        while True: #loop until condition is met
            theta = np.random.uniform(0, 2*np.pi) # generating angle theta value
            phi = np.random.uniform(0, np.pi) #generating angle phi value
            stdx = np.random.normal(0, 0.1) #gaussian noise in x direction
            stdy = np.random.normal(0, 0.3) #gaussian noise in y direction
            x = (2-Zinput)*np.tan(theta)*np.cos(phi)  + stdx #generating random x values
            y = (2-Zinput)*np.tan(theta)*np.sin(phi) + stdy #generaing random y values
            if abs(x) < 0.5 and abs(y) < 0.5: #accept condition
                return x, y
        

def plot(X):
    """Generalised plotting function, takes in X as array to plot"""
    plt.plot(X, 'bo') #normal plot
    plt.title("Plot of generated points") #insert title
    plt.xlabel("Number of points") #label x axis
    plt.ylabel("Value of generated particles") #label y axis
    plt.show() #plot graph
    plt.hist(X, bins = 100) #histogram plot
    plt.title("Histogram of generated points") #insert title
    plt.xlabel("Range of generated points") #label x axis
    plt.ylabel("Frequency of bins") #label y axis
    plt.show() #plot graph

def task1a(n):
    """runs uniform distribution"""
    a = uniformdist(n)
    return a
    
def task1b(n):
    """runs analytical algorithm"""
    a = task1a(n) #generates values from uniform distribution
    b = analytical(a) #runs analytical algorithm
    return b

def task1c(n):
    """runs accept-reject algorithm"""
    A = acceptroutine(n)
    return A

def difference(A, F, n):
    """function to test for percentage difference between 2 arrays, A and F. n indicates size of array"""
    Y = np.zeros(n) #define empty matrix 
    for i in range(n): #for j in range n
        Y[i] = abs(F[i]-A[i])*100 # find percentage difference in values between 2 matrices
    Y = Y[~np.isnan(Y)] #returns all non NA values
    Tot = np.average(Y) #find average of all elements in a matrix
    return Tot # returns average

def truedist():
    """creates an array of true sin distribution"""
    X = np.linspace(0, np.pi, 100) #creates an array of values from 0 to pi with 100 intervals, 100 being the bin size in other algorithms as well
    true = np.sin(X) #returns a true sin distribution
    return X, true

def task1d(A):
    """plots the values of the bins of the histogram of an array (A) and returns its percentage error"""
    h, v = np.histogram(A, bins = 100) #converts array a into histogram and returns value of bins
    max = np.amax(h) #find maximum value of bins
    norm = h/max #normalise array A
    X, true = truedist() # runs true distribution function
    plt.plot(X, norm) #plots a function of the values of the bins of the normalised histogram
    plt.xlabel("Value of angle from 0 to pi")
    plt.ylabel("Value of sin function")
    plt.title("Graph of midpoints of each histogram bin using Monte Carlo Method")
    plt.show()
    plt.plot(X, true) #plots a function of the values of the true sin distribution
    plt.xlabel("Value of angle from 0 to pi")
    plt.ylabel("Value of sin function")
    plt.title("Graph of true sin distribution")
    plt.show()
    error = difference(true, norm, 100) #run percentage error algorithm
    print("Percentage error:", error,"%")

def task1e(n):
    """function to test how percentage error varies with an increasing number of points, n"""
    d = 100 #number of trials
    AY = [] #empty list
    BY = [] #empty list
    Astd = [] #empty list
    Bstd = [] #empty list
    AYerror = np.zeros(d) #empty array to record errors for each trial
    Alogerror = np.zeros(10) #empty array to record errors for each trial
    Blogerror = np.zeros(10) #empty array to record errors for each trial
    BYerror = np.zeros(d) #empty array to record errors for each trial
    xaxis = [] #empty list

    for i in tqdm(range(n)):#tqdm is a progress bar, loop through number of points
        if i % (n/10) == 0: #condition that creates 10 data points
            for t in range(100): #loop through 100 iterations for each data point
                A = task1b(i) #return array for analytical method
                B = task1c(i) #return array for accept-reject method
                h, v = np.histogram(A, bins = 100) #return values of bins of histogram for analytical method
                c, d = np.histogram(B, bins = 100) #return values of bins of histogram for accept-reject method
                Amax = np.amax(h) #returns maximum value of bins for analytical histogram
                Bmax = np.amax(c)#returns maximum value of bins for accept-reject histogram
                Anorm = h/Amax #normalise analytical array
                Bnorm = c/Bmax #normalise accept-reject array
                X, true = truedist() #rturns true sin distribution
                Aerror = difference(true, Anorm, 100) #runs percentage error algoithm for analytical method
                Berror = difference(true, Bnorm, 100) #runs percentage error algoithm for accept-reject method
                AYerror[t] = Aerror #record values of analytical error for each trial
                BYerror[t] = Berror #record values of accept-reject error for each trial
                
            
            AAverage = np.average(AYerror) #calculates average error
            AY.append(AAverage) #appends average error to list
            Avariance = np.std(AYerror) #calculates standard deviation of AYerror array
            Astd.append(Avariance) #appends standard deviation to list
            BAverage = np.average(BYerror) #calculates average error
            BY.append(BAverage) #appends average error to list
            Bvariance = np.std(BYerror) #calculates standard deviation of AYerror array
            Bstd.append(Bvariance) #appends standard deviation to list
            xaxis.append(i)#appends nth iteration to list
    
    Alog = np.log(AY) #log values to linearise plot
    Blog = np.log(BY) #log values to linearise plot
    
    for i in range(10):
        Alogerror[i] = Astd[i]/AY[i] #error propogation of logarithm
        Blogerror[i] = Bstd[i]/BY[i] #error propogation of logarithm
        
    Alogerror = np.asarray(Alogerror) #convert list to array
    Blogerror = np.asarray(Blogerror) #convert list to array
    
    Alog = Alog[~np.isnan(Alog)] #returns all non NA values
    Blog = Blog[~np.isnan(Blog)] #returns all non NA values
    Alogerror = Alogerror[~np.isnan(Alogerror)] #returns all non NA values
    Blogerror = Blogerror[~np.isnan(Blogerror)] #returns all non NA values
        
    Acoeff, Ar, d, e, f = np.polyfit(xaxis[1:], Alog, 1, full=True, w=Alogerror)  #runs a linear fit with error bars, returns coefficients(ACoeff) and average residuel (Ar). All other returned variables are ignored.
    Bcoeff, Br, d, e, f = np.polyfit(xaxis[1:], Blog, 1, full=True, w=Blogerror) #runs a linear fit with error bars, returns coefficients(BCoeff) and average residuel(Br). All other returned variables are ignored.

    m, c, Ar_value, p_value, Astd_err = stats.linregress(xaxis[1:], Alog) #runs linear fit without error bars but returns correlational coefficient (Ar_value). All other returned variables are ignored
    m, c, Br_value, p_value, Bstd_err = stats.linregress(xaxis[1:], Blog) #runs linear fit without error bars but returns correlational coefficient (Ar_value). All other returned variables are ignored
    
    plt.plot(xaxis, AY, 'b', label = "Analytical Method") #plot values for analytical method
    plt.plot(xaxis, BY, 'g', label = "Accept-reject method") #plot values for accept-reject method
    plt.errorbar(xaxis, AY, yerr=Astd, linestyle='--', ecolor = 'b', barsabove = True, capsize=5, ) #plot error bar
    plt.errorbar(xaxis, BY, yerr=Bstd, linestyle='--', ecolor = 'g', barsabove = True, capsize=5, ) #plot error bar
    plt.xlabel("Number of points, n") #label x axis
    plt.ylabel("Percentage accuracy, %") #label y axis
    plt.title("Graph of percentage accuracy vs number of points") #insert title
    plt.legend(loc = 'upper right') #plot legend
    plt.show()
    
    Aslope = Acoeff[0] #extracting float value from list
    Bslope = Bcoeff[0] #extracting float value from list
    Aintercept = Acoeff[1] #extracting float value from list
    Bintercept = Bcoeff[1] #extracting float value from list

    def function(m, c, x):
        """function to return values of the linear fit from the calculated coefficients"""
        y = np.zeros(10) #empty array of y values
        for i in range(10):
            y[i] = m*x[i] + c #formula for linear fit
        return y
    
    Afunction = function(Aslope, Aintercept, xaxis) #creating list of y values for plotting purposes
    Bfunction = function(Bslope, Bintercept, xaxis) #creating list of y values for plotting purposes
    
    plt.plot(xaxis, Afunction, 'b', label="Analytical Method")
    plt.plot(xaxis, Bfunction, 'g', label="Accept-reject method")
    plt.errorbar(xaxis[1:], Alog, yerr=Alogerror, fmt='None', ecolor = 'b', barsabove = True, capsize=5, ) #plot error bar
    plt.errorbar(xaxis[1:], Blog, yerr=Blogerror, fmt='None', ecolor = 'g', barsabove = True, capsize=5, ) #plot error bar
    plt.xlabel("Number of points, n") #label x axis
    plt.ylabel("Log of Percentage accuracy") #label y axis
    plt.title("Linearized graph of percentage accuracy vs number of points") #insert title
    plt.legend(loc = 'upper right') #plot legend
    plt.show()
    
    print("The linear fit for the ANALYTICAL method is of the equation y = ", format(Aslope, '.6f'),"x + ", format(Aintercept, '.6f'))
    print("The value of the correlation coefficient for this method is", format(Ar_value, '.2f'))
    print("The average value of the Residuals of the least-squares fit is ", format(Ar[0], '.4f'))
    print("")
    print("")
    print("The linear fit for the ACCEPT-REJECT method is of the equation y = ", format(Bslope, '.6f'),"x + ", format(Bintercept, '.6f') )
    print("The value of the correlation coefficient for this method is", format(Br_value, '.2f'))
    print("The average value of the Residuals of the least-squares fit is ",format(Br[0], '.4f'))

def task1f(n):
    """function to test how time taken/performance varies with an increasing number of points, n"""
    d = 10 #number of trials
    AY = [] #empty list
    BY = [] #empty list
    AYtime = np.zeros(d) #empty array to record errors for each trial
    BYtime = np.zeros(d) #empty array to record errors for each trial
    Astd = [] #empty list
    Bstd = [] #empty list
    X = [] #empty list
    for i in tqdm(range(n)): #tqdm is a progress bar, loop through number of points
        if i % (n/20) == 0:  #condition that creates 20 data points
            for t in range(10): #loop through 10 iterations for each data point
                Astart = time.time() #record start time for analytical method
                A = task1b(i) #run analytical method
                Aend = time.time() #record end time for analytical method
                Bstart = time.time() #record start time for accept-reject method
                B = task1c(i) #run accept-reject method
                Bend = time.time() #record end time for accept-reject method
                AYtime[t] = Aend - Astart #record values of time taken for analytical method for each trial
                BYtime[t] = Bend - Bstart #record values of time taken for accept-reject method for each trial

            AAverage = np.average(AYtime) #calculates average error
            AY.append(AAverage)#appends average error to list
            BAverage = np.average(BYtime) #calculates average error
            BY.append(BAverage)#appends average error to list
            Avariance = np.std(AY) #calculates standard deviation
            Astd.append(Avariance) #appends standard deviation to list
            Bvariance = np.std(BY)#calculates standard deviation
            Bstd.append(Bvariance) #appends standard deviation to list
            X.append(i) #appends nth iteration to list

    plt.plot(X, AY, 'b',  label = "Analytical Method") #plot values for analytical method
    plt.plot(X, BY, 'g', label = "Accept-reject method") #plot values for accept-reject method
    plt.errorbar(X, AY, yerr=Astd, linestyle='--', ecolor = 'b', barsabove = True, capsize=5, ) #plot error bars
    plt.errorbar(X, BY, yerr=Bstd, linestyle='--', ecolor = 'g', barsabove = True, capsize=5, ) #plots error bars
    plt.xlabel("Number of points, n") #label x axis
    plt.ylabel("Time taken, s") #label y axis
    plt.title("Graph of time taken vs number of points") #insert title
    plt.legend(loc = 'upper left') #plot legend
    plt.show()
    
def plotkernel(n):
    """function to plot a histogram showing the 2D gaussian blur added due to resolution uncertainty"""
    x = np.random.normal(0, 0.1, n) #gaussian noise in x direction
    y = np.random.normal(0, 0.3, n) #gaussian noise in y direction
    plt.hist2d(x, y, bins = 25) #plot 2d histogram
    plt.xlabel("X direction of detector array, m") #label x axis
    plt.ylabel("Y direction of detector array, m") #label y axis
    plt.colorbar()
    plt.title("Contour Plot of Implemented Gaussian Blur")
    plt.show()
    
def task2(mark, N0):
    """function to model a simulated nuclear decay physics experiment. N0 refers to the number of particles, and mark refers to a selection of whether to run the experiment with or without uncertainty"""
    X = np.zeros(N0) #creates empty array of size N0
    Y = np.zeros(N0)#creates empty array of size N0
    Z = np.zeros(N0)#creates empty array of size N0

    for i in tqdm(range(N0)): #tqdm is a progress bar, loop through number of points
        Z[i] = acceptreject(2, 0) #runs modified accept-reject method and fills Z array up with positions of radioactive decay along z direction
        X[i], Y[i] = acceptreject(mark, Z[i])  #runs modified accept-reject method and fills X,Y array up with positions of particles detected on the detector

    if mark == 4:
        plotkernel(N0) #plots gaussian noise function
    
    plt.hist(Z, bins = 50)
    plt.xlabel("Z direction, m") #label x axis
    plt.ylabel("Number of occurences/particles") #label y axis
    plt.title("Histogram plot of particles along the z direction") # insert title
    plt.show()
    
    plt.hist(X, bins = 50)
    plt.xlabel("X direction of detector, m") #label x axis
    plt.ylabel("Number of occurences/particles") #label y axis
    plt.title("Histogram plot of particles along the x direction") # insert title
    plt.show()
    
    plt.hist(Y, bins = 50)
    plt.xlabel("Y direction of detector, m") #label x axis
    plt.ylabel("Number of occurences/particles") #label y axis
    plt.title("Histogram plot of particles along the y direction") # insert title
    plt.show()
    
    plt.scatter(X, Y, s = 1)
    plt.xlabel("X direction of the detector array, m") #label x axis
    plt.ylabel("Y direction of the detector array, m") #label y axis
    plt.title("Scatter plot of detector array") # insert title
    plt.show()
    
    def histogramanalysis(X):
        """function to extract mean and standard deviation of histogram"""
        xvals, xbins = np.histogram(X) #extract values of bins and bin location from histogram
        a = np.shape(xbins) #extract number of bins
        v = a[0] #extract int from tuple
        xmids = np.zeros(v-1) #create empty array of size number of bins -1
        for i in range(v-1):
            xmids[i] = (xbins[i] + xbins[i+1])/2 #caculate midpoint of each bin
        xmean = np.average(xmids, weights = xvals) #find the mean of the bins, with added weight set as value of bins
        xvar = np.average((xmids)**2, weights=xvals) - xmean**2 #variance formula
        xstd = np.sqrt(xvar) #calculate standard deviation
        return xmean, xstd
    
    xmean, xstd = histogramanalysis(X)
    ymean, ystd = histogramanalysis(Y)
    
    print("The mean of the histogram of particles along the x direction is ", format(xmean, '.2f'), "with standard deviation as", format(xstd, '.2f'))
    print("The mean of the histogram of particles along the y direction is ", format(ymean, '.2f'), "with standard deviation as", format(ystd, '.2f'))
    
def confidence(A, n):
    """function to calculate confidence level of cross-sectional level"""
    B = np.zeros(n) #create empty array of size n
    for i in range(n):
        if A[i] >= 5: #condition for significance
            B[i] = 1 #mark particle as observed
            
    v = np.count_nonzero(B == 1) #count number of occurences of particle observed
    v = v/n*100 #percentage of confidence level
    return v
            
def task3a(n):
    """function to simulate collider experiment. N refers to number of particles"""
    D = np.zeros(n) #create empty matrix of size n
    percent = [] #empty list
    x = [] #empty list
    list = [] #empty list

    for sigma in tqdm(range(300)): #looping through cross-sectional areas
        sigma = sigma/200 #loop interval in decimal places
        for i in range(n): #looping through a number of particles
            true = np.random.normal(5.7, 0.4**2) #defining theoretical normal distribution
            Trandom = np.random.poisson(true) #obtaining a sample from a poisson distribution
            while True: #loop until until a positive value of L is obtained to prevent Poisson sampling a negative value
                L = np.random.normal(12, 0.01) #sampling from a normal distribution of luminosity
                if L > 0: #condition if Luminosity is positive
                    break #breaks loop
            Erandom = np.random.poisson(L*sigma) #sampling from a poisson distribution of number of events (X = L*sigma)
            D[i] = Trandom + Erandom #adding both theoretical and experimental readings
        p = confidence(D, n) #runs confidence function
        if p > 95: #break condition
            list.append(sigma)
        percent.append(p) #appends percentage values
        x.append(sigma) #appends cross-sectional values

    plt.scatter(x, percent, s = 2) #produces a scatter plot
    plt.xlabel("Cross-Sectional Values, m^2") #label x axis
    plt.ylabel("Percentage of confidence, %") #label y axis
    plt.title("Graph of percentage confidence level versus cross-sectional area")
    plt.show()
    print("The 95% confidence limit for cross-sectional area was found to be:", list[0], "m^2")
    return list[0]


def task3b(mark ,n):
    """function to determine how uncertainty affects critical cross value. Mark refers to a choice of uncertainty in luminosity or theoretical bg distribution, whereas n refers to number of points."""
    D = np.zeros(n) #create empty matrix of size n
    T = np.zeros(400) #create empty matrix of size 400
    crosssection = [] #empty list
    sigmaerror = [] #empty list
    s = 5 #number of trials
    confidencevalue = np.zeros(s)  #create empty matrix of size s
    x = [] #empty list
    
    if mark == 0: #condition to test uncertainty in luminosity
        start = 0.02*12*100 #minimum value of standard deviation - 2%
        end = 0.05*12*100 #maximum value of standard deviation - 5%
    elif mark == 1: #condition to test uncertainty in background distribution
        start = 0.02*5.7*100 #minimum value of standard deviation - 2%
        end = 0.05*5.7*100 #maximum value of standard deviation - 5%
    
    trial = end - start #interval between start and end
    start = int(start)
    end = int(end)
    trial = int(trial/5) #set number of intervals/data points to be 5

    for u in tqdm(range(start, end, trial)): #loop through uncertainty with progress bar
        u = u/100 #standard deviation value
        if mark == 0: #condition to test uncertainty in luminosity
            ul = u #ul is standard deviation of luminosity
            ut = 0.4
        elif mark == 1: #condition to test uncertainty in background distribution
            ul = 0.1
            ut = u #ut is standard deviation of background distribution
        for t in range(s): #loop through number of trials
            T = np.zeros(400) #reset array T for each iteration
            for sigman in range(0, 300): #looping through cross-sectional areas
                sigma = sigman/200 #loop interval in decimal places
                for i in range(n): #looping through a number of particles
                    while True: #loop until until a positive value of L is obtained to prevent Poisson sampling a negative value
                        true = np.random.normal(5.7, ut**2) #defining theoretical normal distribution
                        if true > 0: #condition if Luminosity is positive   
                            break #breaks loop
                    Trandom = np.random.poisson(true) #obtaining a sample from a poisson distribution
                    while True: #loop until until a positive value of L is obtained to prevent Poisson sampling a negative value
                        L = np.random.normal(12, ul**2) #sampling from a normal distribution of luminosity
                        if L > 0: #condition if Luminosity is positive   
                            break #breaks loop
                    Erandom = np.random.poisson(L*sigma) #sampling from a poisson distribution of number of events (X = L*sigma)
                    D[i] = Trandom + Erandom #adding both theoretical and experimental readings
                p = confidence(D, n) #runs confidence function

                if 95 <= p <= 95.5:  #catch values that are close to cross-sectional limit
                    T[sigman] = sigma #assign these values to T array

            z = T[np.nonzero(T)] #removes all zero values of T
            if len(z) == 0: #catch instance when length of z is 0
                continue #skip
            v = np.amin(z) #find smallest cross-sectional value (critical)
            confidencevalue[t] = v #assign critical cross-sectional value to array

        conf = np.average(confidencevalue[np.nonzero(confidencevalue)]) #find average of critical cross-sectional values
        crosssection.append(conf) #appends average
        std = np.std(confidencevalue[np.nonzero(confidencevalue)]) #find standard deviation  of critical cross-sectional values
        sigmaerror.append(std) #appends standard deviation

        x.append(u) #appends standard deviation


    plt.plot(x, crosssection) #plot points
    plt.errorbar(x, crosssection, yerr=sigmaerror, linestyle='None', marker='.', capsize = 5) #plot error bars

    plt.xlabel("Standard deviation") #label x axis
    plt.ylabel("Critical cross-sectional values, m^2") #label y axis
    plt.title("Graph of critical cross-sectional values versus uncertainty(standard deviation)")
    plt.show()
    

def testinput(e, mini, maxi):
    """input testing function"""
    print("Please enter the value of", e, ". Values of ", e, " must be more than ", mini, " and less than ", maxi, ".", e, ":")
    while True: #loop until true
        try:
            e = int(input()) #input
            while e>maxi or e<mini: #catching values out of range
                print("Value is out of bound. Values must be more than ", mini, " and less than ", maxi)
                e = int(input())
            return e
            break
        except ValueError: #for non-integers
            print ("Invalid value, please enter a number")     
        except: #for any other unknown errors
            print ("Unknown error, please try again.")    
            
MyInput = '0'
while MyInput != 'q':
    MyInput = input('Welcome. \nEnter 1a to view the initial uniform distribution. \nEnter 1b to view the results of the analytical method. \nEnter 1c to view the results of the accept-reject method. \nEnter 1d  view a comparison of the accuracy of results from the analytical or accept-reject method. \nEnter 1e to test how percentage error varies with number of iterations.  \nEnter 1f to test how both methods perform against number of iterations. \nEnter 2a to recreate a nuclear physics experiment. \nEnter 2b to recreate a more realistic nuclear physics experiment with added uncertainties. \nEnter 3a to to recreate a collider experiment, where the cross-sectional area limit is set at 95% confidence level. \nEnter 3b to test how uncertainty in luminosity affects the critical cross-sectional limit. \nEnter 3c to test how uncertainty in theoretical background distribution affects the critical cross-sectional limit')
    print('You entered the choice: ',MyInput)
    if MyInput == '1a': 
        print('You have chosen Task 1a, which is to view the initial uniform distribution.')
        print('The values that you are required to input are: \nn = number of points')
        n = testinput('n', 0, 100000) #input testing function
        a = task1a(n) #generates uniform distribution
        plot(a) #plotting function
    elif MyInput == '1b': 
        print('You have chosen Task 1b, which is to view the results of the analytical method.')
        print('The values that you are required to input are: \nn = number of points')
        n = testinput('n', 0, 100000) #input testing function
        a = task1b(n)
        plot(a) #plotting function
    elif MyInput == '1c': 
        print('You have chosen Task 1c, which is to view the results of the accept-reject method.')
        print('The values that you are required to input are: \nn = number of points')
        n = testinput('n', 0, 100000) #input testing function
        a = task1c(n)
        plot(a) #plotting function
    elif MyInput == '1d': 
        print('You have chosen Task 1d, which is to view a comparison of the accuracy of results from the analytical or accept-reject method.')
        print('The values that you are required to input are: \nn = Number of points \nd = Choice of algorithm. Enter 0 for the analytical method or 1 for the accept-reject method')
        n = testinput('n', 0, 100000) #input testing function
        d = testinput('d', 0, 1) #input testing function
        if d == 0:
            a = task1b(n)
        elif d == 1:
            a = task1c(n)
        task1d(a)
    elif MyInput == '1e': 
        print('You have chosen Task 1e, which is to test how percentage error varies with number of points')
        print('The values that you are required to input are: \nn = number of points')
        n = testinput('n', 0, 100000) #input testing function
        task1e(n)
    elif MyInput == '1f': 
        print('You have chosen Task 1f, which is to test how both methods perform with respect to number of iterations')
        print('The values that you are required to input are: \nn = number of points')
        n = testinput('n', 0, 100000) #input testing function
        task1f(n)
    elif MyInput == '2a': 
        print('You have chosen Task 2a, which is to recreate a nuclear decay physics experiment')
        print('The values that you are required to input are: \nn = number of points. Recommended number = 10000')
        n = testinput('n', 0, 100000) #input testing function
        task2(3, n)
    elif MyInput == '2b': 
        print('You have chosen Task 2b, which is to recreate a more realistic nuclear physics experiment with added uncertainties.')
        print('The values that you are required to input are: \nn = number of points.  Recommended number = 10000')
        n = testinput('n', 0, 100000) #input testing function
        task2(4, n)
    elif MyInput == '3a': 
        print('You have chosen Task 3a, which is to recreate a collider experiment, where the cross-sectional area limit is set at 95% confidence level')
        print('The values that you are required to input are: \nd = Choice of computational requirement. For a computationally cheap but less accurate result, Enter 0. For a computationally expensive but more accurate result, Enter 1.')
        d = testinput('d', 0, 1) #input testing function
        if d == 0:
            task3a(1000)
        elif d == 1:
            task3a(10000)
    elif MyInput == '3b': 
        print('You have chosen Task 3b, which is to test how uncertainty in luminosity affects the critical cross-sectional limit')
        print('The values that you are required to input are: \nd = Choice of computational requirement. For a computationally cheap but less accurate result , Enter 0. For a computationally expensive but more accurate result (takes around 8 minutes), Enter 1.')
        d = testinput('d', 0, 1) #input testing function
        if d == 0:
            task3b(0, 1000)
        elif d == 1:
            task3b(0, 10000)
    elif MyInput == '3c': 
        print('You have chosen Task 3c, which is to test how uncertainty in theoretical background distribution affects the critical cross-sectional limit')
        print('The values that you are required to input are: \nd = Choice of computational requirement. For a computationally cheap but less accurate result, Enter 0. For a computationally expensive but more accurate result (takes around 8 minutes), Enter 1.')
        d = testinput('d', 0, 1) #input testing function
        if d == 0:
            task3b(1, 1000)
        elif d == 1:
            task3b(1, 10000)

        