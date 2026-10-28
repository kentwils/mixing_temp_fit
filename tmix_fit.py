import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# pip install uncertainties, if needed
try:
    import uncertainties.unumpy as unp
    import uncertainties as unc
except:
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
    pipmain(['install','uncertainties'])
    import uncertainties.unumpy as unp
    import uncertainties as unc

#CHANGE THESE VALUES
#inital guess for fitting tmix, set to around where you think the tmix is
tmixguess=34
#data file, must be .csv with columns "T","Mixed","Sep"
inputfile = "example.csv"
#temp range to plot
temp_low = 20
temp_high = 44
#Toggle confidence/prediction bands
show_confidence = True
show_prediction = False

#Define sigmoid
def f(x, x0, k, y0):
    return y0*(1-(1 / (1 + np.exp(-(x-x0)/k))))

#Function to calculate prediction band
def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

#Initialize plot, change some parameters
arialfont = {'fontname':'Arial'}
plt.rcParams.update({'font.size': 32})
plt.rcParams["figure.figsize"] = [10.00, 6.00]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 2
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.tick_params('both', length=10, width=4, which='major')
plt.ylim(0,100)
plt.xlim(temp_low,temp_high)

# import data
columns = ["T", "Mixed", "Sep"]
df = pd.read_csv(inputfile, usecols=columns)
df = df.astype({"T": "float", "Mixed": "float", "Sep": "float"})
df = df.assign(fraction=df.Sep * 100 / (df.Mixed + df.Sep))
x = df['T'].values
y = df['fraction'].values
n = len(y)

#Fit curve. sigmas = standard devs.
popt, pcov = curve_fit(f, x, y,p0=[tmixguess,0.5,100], bounds=([-np.inf,0,0],[np.inf,np.inf,100]))
sigmas = np.sqrt(np.diagonal(pcov))

# retrieve fit parameter values
a = popt[0] #Tmix
b = popt[1] #Width (k)
c = popt[2] #Max phase sep


# compute r^2
r2 = 1.0-(sum((y-f(x,a,b,c))**2)/((n-1.0)*np.var(y,ddof=1)))


# calculate parameter confidence interval
a,b,c = unc.correlated_values(popt, pcov)

print('T_mix: ' + str(a))
print('Width: ' + str(b))
print('Max phase sep: ' + str(c))
print('R^2: ' + str(r2))

# plot data
ax1.scatter(x, y, color='black',label='Data')

# calculate regression confidence interval
px = np.linspace(temp_low, temp_high, 100)
py = c*(1-(1 / (1 + unp.exp(-(px-a)/b))))
nom = unp.nominal_values(py)
std = unp.std_devs(py)


lpb, upb = predband(px, x, y, popt, f, conf=0.95)

# plot the regression
ax1.plot(px, nom, c='blue', label='y=a exp(b x) + c', zorder=0)

# uncertainty region (95% confidence)
if(show_confidence):
    ax1.fill_between(px,nom+1.96*std,nom-1.96*std, alpha=.1, color="blue")

# prediction band (95% confidence)
if(show_prediction):
    ax1.plot(px, lpb, 'k--',label='95% Prediction Band')
    ax1.plot(px, upb, 'k--')

#Plot other lines
ax1.errorbar([popt[0]],[popt[2]/2], xerr=2*sigmas[0], color="blue", fmt="none",capsize=5,markeredgewidth=2)
ax1.axvline(x=popt[0], ymax = (popt[2]/2)/100, linewidth=2, color = "blue", linestyle = "dashed", alpha=1)
ax1.axvline(x=popt[0]+2*sigmas[0], ymax=(popt[2] / 2) / 100, linewidth=2,color = "blue", linestyle="dashed", alpha=.5)
ax1.axvline(x=popt[0]-2*sigmas[0], ymax=(popt[2] / 2) / 100, linewidth=2, color = "blue", linestyle="dashed", alpha=.5)

#Show figure
plt.show()
