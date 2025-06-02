# basic libs
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns

# stats libs
import statsmodels.api as sm
import statsmodels.regression.linear_model as lm
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)



# Load the datasets
cpi = pd.read_excel('YOUR_PATH/CPIAUCSL.xlsx', parse_dates=True, index_col=0)
indpro = pd.read_excel('YOUR_PATH/INDPRO.xlsx', parse_dates=True, index_col=0)
sp500 = pd.read_excel('YOUR_PATH/PerformanceGraphExport.xlsx', sheet_name = "DATA", parse_dates=True, index_col=0)
t10y2y = pd.read_excel('YOUR_PATH/T10Y2Y.xlsx', sheet_name = "DATA", parse_dates=True, index_col=0)
vix = pd.read_csv('YOUR_PATH/VIXCLS.csv', parse_dates=True, index_col=0)
unemployment = pd.read_csv('YOUR_PATH/UNRATE.csv', parse_dates=True, index_col=0)
fed_funds_rate = pd.read_csv('YOUR_PATH/FEDFUNDS.csv', parse_dates=True, index_col=0)

# Resample data to monthly frequency
cpi_monthly = cpi.resample('M').last()
indpro_monthly = indpro.resample('M').last()
sp500_monthly = sp500.resample('M').last()
t10y2y_monthly = t10y2y.resample('M').last()
vix_monthly = vix.resample('M').last()
unemployment_monthly = unemployment.resample('M').last()
fed_funds_rate_monthly = fed_funds_rate.resample('M').last()



# Merge the datasets on the date index with an inner join
merged_data = sp500_monthly.join([cpi_monthly, indpro_monthly, t10y2y_monthly, vix_monthly, unemployment_monthly, fed_funds_rate_monthly], how='inner')

# Ensure VIXCLS is numeric
merged_data['VIXCLS'] = pd.to_numeric(merged_data['VIXCLS'], errors='coerce')

# Drop any rows where VIXCLS is NaN (in case there were non-numeric values)
merged_data = merged_data.dropna(subset=['VIXCLS'])

print(merged_data.head(20))


# Check for missing values
print(merged_data.isnull().sum())

# Plot each column to a separate line plot graph
plt.figure(figsize=(12, 8))

for column in merged_data.columns:
    plt.plot(merged_data.index, merged_data[column], label=column)
    plt.xlabel('Year')
    plt.title("Evolution of "+ column + " over time")
    plt.show()






# Check for stationarity using ADF test
for column in merged_data.columns:
    result = adfuller(merged_data[column])
    print(f'ADF Test for {column}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print('Is Stationary:', result[1] < 0.05)
    print("----------------------------------")
    print('\n')





# Apply logarithm to SP500, Commodity Price Index, Industrial Production
merged_data['ln_SP500'] = np.log(merged_data['SP500'])
merged_data['ln_CPI'] = np.log(merged_data['CPIAUCSL'])
merged_data['ln_INDPRO'] = np.log(merged_data['INDPRO'])
merged_data['ln_FEDFUNDS'] = merged_data['FEDFUNDS'].diff()


# Calculate returns/growth rates by differencing the logged series
merged_data['D_ln_SP500'] = merged_data['ln_SP500'].diff()
merged_data['D_ln_CPI'] = merged_data['ln_CPI'].diff()
merged_data['D_ln_INDPRO'] = merged_data['ln_INDPRO'].diff()
merged_data['D_T10Y2Y'] = merged_data['T10Y2Y'].diff()
merged_data['D_UNRATE'] = merged_data['UNRATE'].diff()
merged_data['D_ln_FEDFUNDS'] = merged_data['ln_FEDFUNDS'].diff()


pd.set_option("display.max_columns", None)
merged_data = merged_data.dropna()





# Check for stationarity using ADF test
for column in merged_data.columns:
    result = adfuller(merged_data[column])
    print(f'ADF Test for {column}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print('Is Stationary:', result[1] < 0.05)
    print("----------------------------------")
    print('\n')




#%%%%%%


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm


# Select the variables
Y = merged_data['D_ln_SP500']
X = merged_data[['VIXCLS' , 'D_T10Y2Y', 'D_ln_FEDFUNDS', 'D_UNRATE']]
s_t = merged_data['D_ln_INDPRO']  # Transition variable

# Add a constant term to X
X = sm.add_constant(X)

# Ensure that all data is aligned
Y = Y[X.index]
s_t = s_t[X.index]

# Define the logistic transition function
def logistic_transition(s, gamma, c):
    return 1 / (1 + np.exp(-gamma * (s - c)))

# Define the STR model residuals
def STR_model(params, Y, X, s):
    k = X.shape[1]  # Number of regressors including constant
    beta0 = params[:k]
    beta1 = params[k:2*k]
    gamma = params[-2]
    c = params[-1]
    
    G = logistic_transition(s, gamma, c)
    Y_hat = X.dot(beta0) + G * X.dot(beta1)
    residuals = Y - Y_hat
    return residuals

# Define the objective function (sum of squared residuals)
def objective_function(params, Y, X, s):
    residuals = STR_model(params, Y, X, s)
    return np.sum(residuals**2)

# Initial parameter guesses
k = X.shape[1]
initial_params = np.zeros(2*k + 2)  # beta0, beta1, gamma, c

# Set initial values for gamma and c
initial_params[-2] = 1.0  # gamma (slope of transition function)
initial_params[-1] = s_t.mean()  # c (location parameter)

# Bounds for gamma (positive) and c (within the range of s_t)
bounds = [(-np.inf, np.inf)] * (2*k) + [(0.01, 100), (s_t.min(), s_t.max())]

# Optimize the objective function
result = minimize(
    objective_function,
    initial_params,
    args=(Y, X, s_t),
    bounds=bounds,
    method='L-BFGS-B'
)

# Check if the optimization was successful and print results
if result.success:
    estimated_params = result.x
    print('Optimization was successful.')
    print('Estimated parameters:')
    print('beta0 (linear part coefficients):', estimated_params[:k])
    print('beta1 (nonlinear part coefficients):', estimated_params[k:2*k])
    print('gamma (transition speed parameter):', estimated_params[-2])
    print('c (threshold parameter):', estimated_params[-1])
else:
    print('Optimization failed.')
    print('Reason:', result.message)



import matplotlib.pyplot as plt
import numpy as np

# Extract estimated parameters
beta0 = estimated_params[:k]
beta1 = estimated_params[k:2*k]
gamma = estimated_params[-2]
c = estimated_params[-1]


# Calculate the transition function values
s_sorted = np.sort(s_t)
G_values = logistic_transition(s_sorted, gamma, c)

plt.figure(figsize=(8, 6))
plt.plot(s_sorted, G_values)
plt.title('Estimated Transition Function')
plt.xlabel('D_UNRATE')
plt.ylabel('G(s_t; gamma, c)')
plt.grid(True)
plt.show()





# Compute the transition function values for all observations
G_t = logistic_transition(s_t, gamma, c)

# Compute fitted values
Y_hat = X.dot(beta0) + G_t * X.dot(beta1)

# Plot actual vs fitted values
plt.figure(figsize=(12, 6))
plt.plot(Y.index, Y, label='Actual', marker='o')
plt.plot(Y.index, Y_hat, label='Fitted', linestyle='--', marker='x')
plt.legend()
plt.title('Actual vs Fitted Values of D_ln_SP500')
plt.xlabel('Date')
plt.ylabel('D_ln_SP500')
plt.show()


# Compute residuals
residuals = Y - Y_hat

# Plot residuals over time
plt.figure(figsize=(12, 6))
plt.plot(Y.index, residuals)
plt.title('Residuals of STR Model')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()





from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(residuals, lags=[12], return_df=True)
print('Ljung-Box test for autocorrelation:')
print(lb_test)



from scipy.stats import jarque_bera

jb_stat, jb_pvalue = jarque_bera(residuals)
print('Jarque-Bera test for normality:')
print(f'JB Statistic: {jb_stat}, p-value: {jb_pvalue}')




# For each coefficient, plot the total effect over time
coefficients = {}
for i, var in enumerate(X.columns):
    total_effect = beta0[i] + G_t * beta1[i]
    coefficients[var] = total_effect

plt.figure(figsize=(12, 6))
for var in X.columns[1:]:  # Exclude the constant term
    plt.plot(Y.index, coefficients[var], label=var)
plt.legend()
plt.title('Time-varying Coefficients in STR Model')
plt.xlabel('Date')
plt.ylabel('Coefficient Value')
plt.show()




import statsmodels.api as sm

# Plot ACF
sm.graphics.tsa.plot_acf(residuals, lags=24)
plt.title('Autocorrelation Function (ACF) of Residuals')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()

# Plot PACF
sm.graphics.tsa.plot_pacf(residuals, lags=24)
plt.title('Partial Autocorrelation Function (PACF) of Residuals')
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.show()




# For each coefficient, calculate the total effect over time
coefficients = {}
for i, var in enumerate(X.columns):
    total_effect = beta0[i] + G_t * beta1[i]
    coefficients[var] = total_effect

# Plot the coefficients (excluding the constant term)
plt.figure(figsize=(12, 6))
for var in X.columns[1:]:  # Exclude 'const'
    plt.plot(Y.index, coefficients[var], label=var)
plt.legend()
plt.title('Time-Varying Coefficients in STR Model')
plt.xlabel('Date')
plt.ylabel('Coefficient Value')
plt.show()









#%%%%%


# Define dependent and independent variables
y = merged_data['D_ln_SP500']
X = merged_data[['D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y', 'VIXCLS', 'D_UNRATE', 'D_ln_FEDFUNDS']]

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())




from statsmodels.stats.outliers_influence import variance_inflation_factor


X_vars = merged_data[['D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y', 'VIXCLS', 'D_UNRATE', 'D_ln_FEDFUNDS']]
X_vars = sm.add_constant(X_vars)  # Add constant term

# Calculate VIF for each explanatory variable
vif_data = pd.DataFrame()
vif_data['Feature'] = X_vars.columns
vif_data['VIF'] = [variance_inflation_factor(X_vars.values, i) for i in range(X_vars.shape[1])]

print(vif_data)




# Dependent variable
y = merged_data['D_ln_SP500']

# Independent variables
X = merged_data[['D_ln_FEDFUNDS']]  # Simplified model
X = sm.add_constant(X)

# Transition variable
Z = merged_data['VIXCLS']
Z_squared = Z ** 2
Z_cubed = Z ** 3

# Auxiliary regression
X_aux = X.copy()
X_aux['Z'] = Z
X_aux['Z_squared'] = Z_squared
X_aux['Z_cubed'] = Z_cubed

# Original model
model = sm.OLS(y, X).fit()
SSR0 = model.ssr

# Auxiliary model
model_aux = sm.OLS(y, X_aux).fit()
SSR1 = model_aux.ssr

# Calculate the LST test statistic
LM_stat = len(y) * (SSR0 - SSR1) / SSR0
p_value = stats.chi2.sf(LM_stat, df=3)

print(f'LST Test Statistic: {LM_stat}')
print(f'p-value: {p_value}')


#%%%%%%

# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm

# Ensure 'merged_data' DataFrame is updated with monthly frequency data
# Dependent variable
y = merged_data['D_ln_SP500'].values

# Independent variables
X = merged_data[['const', 'D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y', 'D_UNRATE', 'D_ln_FEDFUNDS']].values

# Exclude 'VIXCLS' from X since it will be used as the transition variable


# Re-prepare X without 'const' and add constant term properly
X_vars = merged_data[['D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y', 'D_UNRATE', 'D_ln_FEDFUNDS']]
X = sm.add_constant(X_vars).values  # This adds the intercept term

# Transition variable
s = merged_data['VIXCLS'].values


















#%%%%%%

# Apply logarithm to SP500, Commodity Price Index, Industrial Production
merged_data['ln_SP500'] = np.log(merged_data['SP500'])
merged_data['ln_CPI'] = np.log(merged_data['CPIAUCSL'])
merged_data['ln_INDPRO'] = np.log(merged_data['INDPRO'])


# # Create differenced variable for T10Y2Y

merged_data['D_ln_SP500'] = merged_data['ln_SP500'].diff()
merged_data['D_ln_CPI'] = merged_data['ln_CPI'].diff()
merged_data['D_ln_INDPRO'] = merged_data['ln_INDPRO'].diff()
merged_data['D_FedFundsRate'] = merged_data['FEDFUNDS'].diff()
merged_data['D_Unemployment'] = merged_data['UNRATE'].diff()
merged_data['D_T10Y2Y'] = merged_data['T10Y2Y'].diff()

pd.set_option("display.max_columns", None)
merged_data = merged_data.dropna()



# Plot each column to a separate line plot graph
plt.figure(figsize=(12, 8))

for column in merged_data.columns:
    plt.plot(merged_data.index, merged_data[column], label=column)
    plt.xlabel('Year')
    plt.title("Evolution of "+ column + " over time")
    plt.show()



# Check for stationarity using ADF test
for column in merged_data.columns:
    result = adfuller(merged_data[column])
    print(f'ADF Test for {column}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print('Is Stationary:', result[1] < 0.05)
    print("----------------------------------")
    print('\n')



# Plotting the logged levels
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(merged_data.index, merged_data['ln_SP500'], label='ln_SP500')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(merged_data.index, merged_data['ln_CPI'], label='ln_CPI', color='orange')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(merged_data.index, merged_data['ln_INDPRO'], label='ln_INDPRO', color='green')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the first differences (returns/growth rates)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(merged_data.index, merged_data['D_ln_SP500'], label='D_ln_SP500')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(merged_data.index, merged_data['D_ln_CPI'], label='D_ln_CPI', color='orange')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(merged_data.index, merged_data['D_ln_INDPRO'], label='D_ln_INDPRO', color='green')
plt.legend()
plt.tight_layout()
plt.show()


# Plotting the transition variable
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(merged_data.index, merged_data['T10Y2Y'], label='T10Y2Y', color='blue')
plt.legend()







# Drop NaNs resulting from differencing
merged_data = merged_data.dropna()

# Perform ADF test on D_T10Y2Y
result = adfuller(merged_data['D_T10Y2Y'])
print('ADF Test for D_T10Y2Y:')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
print('Is Stationary:', result[1] < 0.05)
print("----------------------------------")




# Define dependent and independent variables
y = merged_data['D_ln_SP500']
X = merged_data[['D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y', 'VIXCLS', 'D_Unemployment', 'D_FedFundsRate']]

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())





# Augment the dataset with polynomial terms of the transition variable
Z = merged_data['VIXCLS']
Z_squared = Z ** 2
Z_cubed = Z ** 3

# Define the auxiliary regression variables
X_aux = X.copy()
X_aux['Z'] = Z
X_aux['Z_squared'] = Z_squared
X_aux['Z_cubed'] = Z_cubed

# Fit the auxiliary regression
model_aux = sm.OLS(y, X_aux).fit()

# Calculate the test statistic
SSR0 = model.ssr  # Sum of squared residuals from the original model
SSR1 = model_aux.ssr  # Sum of squared residuals from the auxiliary model
LM_stat = len(y) * (SSR0 - SSR1) / SSR0
p_value = stats.chi2.sf(LM_stat, df=3)  # Degrees of freedom = number of added terms

print(f'LST Test Statistic: {LM_stat}')
print(f'p-value: {p_value}')





# Augment the dataset with polynomial terms of the new transition variable
Z = merged_data['D_T10Y2Y']
Z_squared = Z ** 2
Z_cubed = Z ** 3

# Define the auxiliary regression variables
X_aux = X.copy()
X_aux['Z'] = Z
X_aux['Z_squared'] = Z_squared
X_aux['Z_cubed'] = Z_cubed

# Fit the auxiliary regression
model_aux = sm.OLS(y, X_aux).fit()

# Calculate the test statistic
SSR0 = model.ssr  # Sum of squared residuals from the original model
SSR1 = model_aux.ssr  # Sum of squared residuals from the auxiliary model
LM_stat = len(y) * (SSR0 - SSR1) / SSR0
p_value = stats.chi2.sf(LM_stat, df=3)  # Degrees of freedom = number of added terms

print(f'LST Test Statistic: {LM_stat}')
print(f'p-value: {p_value}')





# Augment the dataset with polynomial terms of the new transition variable
Z = merged_data['D_Unemployment']
Z_squared = Z ** 2
Z_cubed = Z ** 3

# Define the auxiliary regression variables
X_aux = X.copy()
X_aux['Z'] = Z
X_aux['Z_squared'] = Z_squared
X_aux['Z_cubed'] = Z_cubed

# Fit the auxiliary regression
model_aux = sm.OLS(y, X_aux).fit()

# Calculate the test statistic
SSR0 = model.ssr  # Sum of squared residuals from the original model
SSR1 = model_aux.ssr  # Sum of squared residuals from the auxiliary model
LM_stat = len(y) * (SSR0 - SSR1) / SSR0
p_value = stats.chi2.sf(LM_stat, df=3)  # Degrees of freedom = number of added terms

print(f'LST Test Statistic: {LM_stat}')
print(f'p-value: {p_value}')



# Prepare the data
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

# Dependent variable
y = merged_data['D_ln_SP500'].values

# Independent variables
X = merged_data[['D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y', 'VIXCLS', 'D_FedFundsRate']].values
X = sm.add_constant(X)  # Add intercept term

# Transition variable
s = merged_data['D_Unemployment'].values

# Number of independent variables (including constant)
n_samples, k = X.shape

# Define the STR model function
def str_model(params, y, X, s):
    # Number of observations and independent variables
    n_samples, k = X.shape

    # Extract parameters
    beta0 = params[:k]            # Base regime coefficients (shape: (k,))
    beta1 = params[k:2*k]         # Transition regime coefficients (shape: (k,))
    gamma = params[-2]            # Slope parameter
    c = params[-1]                # Location parameter

    # Ensure gamma is positive to maintain identification
    gamma = abs(gamma)

    # Compute the transition function G(s; gamma, c)
    G = 1 / (1 + np.exp(-gamma * (s - c)))  # Shape: (n_samples,)

    # Reshape G to (n_samples, 1) for broadcasting
    G = G.reshape(-1, 1)  # Shape: (n_samples, 1)

    # Compute the coefficients for each observation
    beta = beta0 + beta1 * G  # Shape: (n_samples, k)

    # Predicted y values
    y_pred = np.sum(X * beta, axis=1)  # Shape: (n_samples,)

    # Residuals
    residuals = y - y_pred

    # Sum of squared residuals
    SSR = np.sum(residuals**2)

    return SSR

# Fit the linear model to get initial estimates for beta0
linear_model = sm.OLS(y, X).fit()
initial_beta0 = linear_model.params  # Use params directly

# Initial guesses for beta1, gamma, and c
initial_beta1 = np.zeros(k)  # Start with zeros for beta1

# Initial guesses for gamma and c
initial_gamma = 10.0  # Try a higher starting value
initial_c = np.mean(s)  # Mean of the transition variable

# Combine all initial parameters
initial_params = np.concatenate([initial_beta0, initial_beta1, [initial_gamma, initial_c]])

# Bounds for parameters
bounds = [(-np.inf, np.inf)] * (2 * k) + [(1e-3, 1e3), (np.min(s), np.max(s))]

# Set options for the optimizer
options = {'maxiter': 10000}

# Minimize the SSR with increased maxiter
result = minimize(str_model, initial_params, args=(y, X, s), bounds=bounds, method='L-BFGS-B', options=options)

# Check if optimization was successful
if result.success:
    print("Optimization was successful.")
else:
    print("Optimization failed:", result.message)

# Extract estimated parameters
estimated_params = result.x
print("Estimated Parameters:")
print(estimated_params)



estimated_params = [
    1.44838185e-01,  # beta0[0] (Intercept)
    6.20809026e-02,  # beta0[1] (D_ln_CPI)
    1.13748160e+00,  # beta0[2] (D_ln_INDPRO)
    5.04377848e-02,  # beta0[3] (D_T10Y2Y)
    -6.50337505e-03, # beta0[4] (VIXCLS)
    3.24847572e-02,  # beta0[5] (D_FedFundsRate)
    4.19292786e-01,  # beta1[0] (Intercept)
    -6.52016095e-02, # beta1[1] (D_ln_CPI)
    -4.31194412e-01, # beta1[2] (D_ln_INDPRO)
    -6.49715381e-01, # beta1[3] (D_T10Y2Y)
    -2.55542268e-02, # beta1[4] (VIXCLS)
    -1.18398337e+00, # beta1[5] (D_FedFundsRate)
    1.00091041e+01,  # gamma (Slope parameter)
    3.47767485e-01   # c (Location parameter)
]



hess_inv = result.hess_inv.todense()

# Compute standard errors
standard_errors = np.sqrt(np.diag(hess_inv))

# Print the standard errors
print("Standard Errors:")
print(standard_errors)




from statsmodels.stats.outliers_influence import variance_inflation_factor

# Prepare the independent variables for VIF calculation
variables = pd.DataFrame(X[:, 1:], columns=['D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y', 'VIXCLS', 'D_FedFundsRate'])

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data['Feature'] = variables.columns
vif_data['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

print(vif_data)




# Calculate residuals from the STR model
def calculate_residuals(params, y, X, s):
    n_samples, k = X.shape
    beta0 = params[:k]
    beta1 = params[k:2*k]
    gamma = abs(params[-2])
    c = params[-1]
    G = 1 / (1 + np.exp(-gamma * (s - c)))
    G = G.reshape(-1, 1)
    beta = beta0 + beta1 * G
    y_pred = np.sum(X * beta, axis=1)
    residuals = y - y_pred
    return residuals

residuals = calculate_residuals(estimated_params, y, X, s)

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(merged_data.index, residuals)
plt.title('Residuals from STR Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()



import matplotlib.pyplot as plt

# Compute the transition function G(s; gamma, c)
gamma_est = estimated_params[-2]
c_est = estimated_params[-1]
s_sorted = np.sort(s)
G_est = 1 / (1 + np.exp(-gamma_est * (s_sorted - c_est)))

# Plot the transition function
plt.figure(figsize=(8, 6))
plt.plot(s_sorted, G_est, label='Estimated Transition Function')
plt.xlabel('Change in Unemployment Rate (D_Unemployment)')
plt.ylabel('Transition Function G(s)')
plt.title('Smooth Transition Function')
plt.legend()
plt.grid(True)
plt.show()



















#%%%%%%

#Check for Seasonality

# Plot ACF and PACF for the differenced series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# SP500 Returns
plot_acf(merged_data['D_ln_SP500'], lags=24, title='ACF of D_ln_SP500')
plt.show()
plot_pacf(merged_data['D_ln_SP500'], lags=18, title='PACF of D_ln_SP500')
plt.show()

# Commodity Price Index Growth Rates
plot_acf(merged_data['D_ln_CPI'], lags=24, title='ACF of D_ln_CPI')
plt.show()
plot_pacf(merged_data['D_ln_CPI'], lags=18, title='PACF of D_ln_CPI')
plt.show()

# Industrial Production Growth Rates
plot_acf(merged_data['D_ln_INDPRO'], lags=24, title='ACF of D_ln_INDPRO')
plt.show()
plot_pacf(merged_data['D_ln_INDPRO'], lags=18, title='PACF of D_ln_INDPRO')
plt.show()

# Rate Spread
plot_acf(merged_data['T10Y2Y'], lags=24, title='ACF of D_T10Y2Y')
plt.show()
plot_pacf(merged_data['T10Y2Y'], lags=18, title='PACF of D_T10Y2Y')
plt.show()











#%%%%

# Function to perform ADF test
def adf_test(series, title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series)
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations Used']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_string())
    print('--------------------------------------')

# Perform ADF tests on the differenced variables
adf_test(merged_data['D_ln_SP500'], 'D_ln_SP500')
adf_test(merged_data['D_ln_CPI'], 'D_ln_CPI')
adf_test(merged_data['D_ln_INDPRO'], 'D_ln_INDPRO')
adf_test(merged_data['D_T10Y2Y'], 'D_T10Y2Y')



# Prepare the independent variables (X) and dependent variable (Y)
X = merged_data[['D_ln_CPI', 'D_ln_INDPRO', 'D_T10Y2Y']]
Y = merged_data['D_ln_SP500']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) regression model
linear_model = sm.OLS(Y, X).fit()

# Print the regression results
print(linear_model.summary())


#%%%%

#Preparing the Data for STR Model

# Transition variable
q_t = merged_data['VIXCLS']

# Check stationarity
adf_test(q_t, 'VIXCLS')




# Independent variables (X) and dependent variable (Y)
X = merged_data[['D_ln_CPI', 'D_ln_INDPRO', 'D_FedFundsRate']]
Y = merged_data['D_ln_SP500'].loc[X.index]

# Add a constant to X for the intercept term
X = sm.add_constant(X)

# Fit the OLS model
linear_model = sm.OLS(Y, X).fit()

# Print the summary
print(linear_model.summary())


#Estimating the STR Model Parameters
import numpy as np
from scipy.optimize import minimize

# Define the STR model function
def str_model(params, Y, X, q_t):
    # Number of predictors including constant
    k = X.shape[1]
    
    # Split parameters
    beta = params[:k]        # Linear coefficients
    gamma = params[k:2*k]    # Non-linear coefficients
    alpha = params[-2]       # Smoothness parameter
    c = params[-1]           # Threshold parameter
    
    # Transition function
    G = 1 / (1 + np.exp(-alpha * (q_t - c)))
    
    # Predicted Y
    Y_hat = X @ beta + G * (X @ gamma)
    
    # Residuals
    residuals = Y - Y_hat
    
    # Sum of squared residuals
    SSR = np.sum(residuals**2)
    
    return SSR

# Initial parameter guesses
beta_init = linear_model.params.values
gamma_init = np.zeros_like(beta_init)
alpha_init = 1.0
c_init = q_t.mean()

initial_params = np.concatenate([beta_init, gamma_init, [alpha_init, c_init]])

# Bounds for parameters (optional)
bounds = [(None, None)] * len(initial_params)
bounds[-2] = (0.01, None)  # alpha > 0

# Optimize the STR model
result = minimize(
    str_model,
    initial_params,
    args=(Y.values, X.values, q_t.values),
    method='L-BFGS-B',
    bounds=bounds
)

# Check if optimization was successful
if result.success:
    estimated_params = result.x
    print("Optimization was successful.")
else:
    print("Optimization failed:", result.message)

# Extract estimated parameters
k = X.shape[1]
beta_est = estimated_params[:k]
gamma_est = estimated_params[k:2*k]
alpha_est = estimated_params[-2]
c_est = estimated_params[-1]

print("\nEstimated beta coefficients (linear part):")
print(beta_est)

print("\nEstimated gamma coefficients (non-linear part):")
print(gamma_est)

print(f"\nEstimated alpha (smoothness parameter): {alpha_est}")
print(f"Estimated c (threshold parameter): {c_est}")




#Plotting the Transition Function

import matplotlib.pyplot as plt

# Compute the transition function values
G = 1 / (1 + np.exp(-alpha_est * (q_t - c_est)))

# Plot the transition function
plt.figure(figsize=(10, 6))
plt.plot(merged_data.index, G, label='Transition Function G(q_t)')
plt.xlabel('Date')
plt.ylabel('G(q_t)')
plt.title('Transition Function over Time')
plt.legend()
plt.grid(True)
plt.show()



#Comparing STR Model to Linear Model

# Compute residuals for the linear model
linear_residuals = Y.values - linear_model.fittedvalues.values
linear_SSR = np.sum(linear_residuals**2)

# Compute residuals for the STR model
G = 1 / (1 + np.exp(-alpha_est * (q_t.values - c_est)))
Y_hat_str = X.values @ beta_est + G * (X.values @ gamma_est)
str_residuals = Y.values - Y_hat_str
str_SSR = np.sum(str_residuals**2)

print(f"Linear Model SSR: {linear_SSR:.4f}")
print(f"STR Model SSR: {str_SSR:.4f}")



#Model Diagnostics
#Residual Analysis

# Plot residuals over time
plt.figure(figsize=(10, 6))
plt.plot(merged_data.index, str_residuals, label='STR Model Residuals')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.title('STR Model Residuals Over Time')
plt.legend()
plt.grid(True)
plt.show()


#Check for Autocorrelation

from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(str_residuals)
print(f'Durbin-Watson statistic for STR model residuals: {dw_stat:.4f}')


#Normality of Residuals
#Histogram and Q-Q Plot

import seaborn as sns
import scipy.stats as stats

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(str_residuals, kde=True)
plt.title('Histogram of STR Model Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# Q-Q Plot
plt.figure(figsize=(6, 6))
stats.probplot(str_residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of STR Model Residuals')
plt.show()


#Shapiro-Wilk Test
from scipy.stats import shapiro
shapiro_test = shapiro(str_residuals)
print(f'Shapiro-Wilk Test statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}')


#Heteroscedasticity
#Breusch-Pagan Test

import statsmodels.stats.api as sms

# Explanatory variables for the test (excluding the constant term)
exog = X.values[:, 1:]

bp_test = sms.het_breuschpagan(str_residuals, exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))




# Compute the covariance matrix
cov_matrix = result.hess_inv.todense()
std_errors = np.sqrt(np.diag(cov_matrix))

# Print parameter estimates with standard errors
param_names = ['beta_const', 'beta_D_ln_CPI', 'beta_D_ln_INDPRO', 'beta_D_T10Y2Y',
               'gamma_const', 'gamma_D_ln_CPI', 'gamma_D_ln_INDPRO', 'gamma_D_T10Y2Y',
               'alpha', 'c']

for name, est, se in zip(param_names, estimated_params, std_errors):
    print(f"{name}: Estimate = {est:.4f}, Std. Error = {se:.4f}")




















# Check for stationarity using ADF test
for column in merged_data.columns:
    result = adfuller(merged_data[column])
    print(f'ADF Test for {column}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print('Is Stationary:', result[1] < 0.05)
    print("----------------------------------")
    print('\n')



#%%%%%%

# Apply logarithm to SP500, Commodity Price Index, Industrial Production
merged_data['ln_SP500'] = np.log(merged_data['SP500'])
merged_data['ln_CPI'] = np.log(merged_data['CPIAUCSL'])
merged_data['ln_INDPRO'] = np.log(merged_data['INDPRO'])



# Calculate returns/growth rates by differencing the logged series
merged_data['D_ln_SP500'] = merged_data['ln_SP500'].diff()
merged_data['D_ln_CPI'] = merged_data['ln_CPI'].diff()
merged_data['D_ln_INDPRO'] = merged_data['ln_INDPRO'].diff()
merged_data['D_T10Y2Y'] = merged_data['T10Y2Y'].diff()

pd.set_option("display.max_columns", None)
merged_data = merged_data.dropna()



# Plot each column to a separate line plot graph
plt.figure(figsize=(12, 8))

for column in merged_data.columns:
    plt.plot(merged_data.index, merged_data[column], label=column)
    plt.xlabel('Year')
    plt.title("Evolution of "+ column + " over time")
    plt.show()



# Check for stationarity using ADF test
for column in merged_data.columns:
    result = adfuller(merged_data[column])
    print(f'ADF Test for {column}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print('Is Stationary:', result[1] < 0.05)
    print("----------------------------------")
    print('\n')



# Plotting the logged levels
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(merged_data.index, merged_data['ln_SP500'], label='ln_SP500')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(merged_data.index, merged_data['ln_CPI'], label='ln_CPI', color='orange')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(merged_data.index, merged_data['ln_INDPRO'], label='ln_INDPRO', color='green')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the first differences (returns/growth rates)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(merged_data.index, merged_data['D_ln_SP500'], label='D_ln_SP500')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(merged_data.index, merged_data['D_ln_CPI'], label='D_ln_CPI', color='orange')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(merged_data.index, merged_data['D_ln_INDPRO'], label='D_ln_INDPRO', color='green')
plt.legend()
plt.tight_layout()
plt.show()


# Plotting the transition variable
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(merged_data.index, merged_data['T10Y2Y'], label='T10Y2Y', color='blue')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(merged_data.index, merged_data['D_T10Y2Y'], label='D_T10Y2Y', color='red')
plt.legend()
plt.tight_layout()
plt.show()

#%%%%%%

#Check for Seasonality

# Plot ACF and PACF for the differenced series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# SP500 Returns
plot_acf(merged_data['D_ln_SP500'], lags=24, title='ACF of D_ln_SP500')
plt.show()
plot_pacf(merged_data['D_ln_SP500'], lags=18, title='PACF of D_ln_SP500')
plt.show()

# Commodity Price Index Growth Rates
plot_acf(merged_data['D_ln_CPI'], lags=24, title='ACF of D_ln_CPI')
plt.show()
plot_pacf(merged_data['D_ln_CPI'], lags=18, title='PACF of D_ln_CPI')
plt.show()

# Industrial Production Growth Rates
plot_acf(merged_data['D_ln_INDPRO'], lags=24, title='ACF of D_ln_INDPRO')
plt.show()
plot_pacf(merged_data['D_ln_INDPRO'], lags=18, title='PACF of D_ln_INDPRO')
plt.show()

# Rate Spread
plot_acf(merged_data['D_T10Y2Y'], lags=24, title='ACF of D_T10Y2Y')
plt.show()
plot_pacf(merged_data['D_T10Y2Y'], lags=18, title='PACF of D_T10Y2Y')
plt.show()


#%%%%

# Dependent Variable
y = merged_data['D_ln_SP500']

# Independent Variables
x1 = merged_data['D_ln_CPI']
x2 = merged_data['D_ln_INDPRO']

# Combine independent variables into one DataFrame
X = pd.concat([x1, x2], axis=1)
X.columns = ['D_ln_CPI', 'D_ln_INDPRO']

# Transition Variable
z = merged_data['D_T10Y2Y']

# Add constant term for the intercept
X = sm.add_constant(X)

# Fit the linear model
linear_model = sm.OLS(y, X).fit()

# Print summary of the linear model
print(linear_model.summary())


# Plot residuals
plt.figure(figsize=(10, 5))
plt.plot(linear_model.resid, label='Residuals')
plt.legend()
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 4))
sns.histplot(linear_model.resid, kde=True)
plt.title('Residuals Distribution')
plt.show()


# Durbin-Watson test for autocorrelation
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(linear_model.resid)
print(f'Durbin-Watson statistic: {dw_stat}')

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(linear_model.resid, lags=[12], return_df=True)
print(lb_test)

# Breusch-Pagan test for heteroskedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(linear_model.resid, X)
bp_test_results = dict(zip(['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'], bp_test))
print(bp_test_results)






# Create auxiliary variables for the LM test
def create_auxiliary_variables(X, z):
    """
    Create auxiliary variables by multiplying each regressor with the transition variable
    """
    Z = pd.DataFrame()
    for col in X.columns:
        if col != 'const':
            Z[f'{col}_z'] = X[col] * z
            Z[f'{col}_z2'] = X[col] * z**2
            Z[f'{col}_z3'] = X[col] * z**3
    return Z

# Combine original regressors and auxiliary variables
Z = create_auxiliary_variables(X, z)
X_LM = pd.concat([X, Z], axis=1)

# Fit the auxiliary regression
auxiliary_model = sm.OLS(y, X_LM).fit()

# Compute the LM statistic
LM_stat = (auxiliary_model.rsquared - linear_model.rsquared) * len(y)
p_value = 1 - stats.chisqprob(LM_stat, df=Z.shape[1])

print(f'LM Statistic: {LM_stat}')
print(f'p-value: {p_value}')



#Estimation of the STR Model

#Define the Logistic Transition Function
def logistic_transition(z, gamma, c):
    return 1 / (1 + np.exp(-gamma * (z - c)))

#Define the STR Model
def str_model(params, y, X, z):
    """
    STR model function for parameter estimation
    """
    # Number of regressors including constant
    k = X.shape[1]
    beta_linear = params[:k]            # Linear part coefficients
    beta_nonlinear = params[k:2*k]      # Non-linear part coefficients
    gamma = params[-2]
    c = params[-1]
    
    # Calculate transition function
    G = logistic_transition(z, gamma, c)
    
    # Predicted values
    y_pred = X.dot(beta_linear) + (X.values * G[:, np.newaxis]).dot(beta_nonlinear)
    
    # Residuals
    residuals = y - y_pred
    
    return residuals



# Number of parameters
k = X.shape[1]  # Number of regressors including constant

# Initial parameter estimates
initial_params = np.concatenate([linear_model.params.values, linear_model.params.values, [1.0], [z.mean()]])

print('Initial Parameters:')
print(initial_params)


# Define the objective function (sum of squared residuals)
def objective_function(params, y, X, z):
    residuals = str_model(params, y, X, z)
    return np.sum(residuals**2)

# Bounds and constraints (gamma > 0)
bounds = [(None, None)] * len(initial_params)
bounds[-2] = (0.001, None)  # gamma > 0

# Estimation
result = minimize(objective_function, initial_params, args=(y, X, z.values), bounds=bounds, method='L-BFGS-B')

# Estimated parameters
est_params = result.x

print('Estimated Parameters:')
print(est_params)



# Define the objective function (sum of squared residuals)
def objective_function(params, y, X, z):
    residuals = str_model(params, y, X, z)
    return np.sum(residuals**2)

# Bounds and constraints (gamma > 0)
bounds = [(None, None)] * len(initial_params)
bounds[-2] = (0.001, None)  # gamma > 0

# Estimation
result = minimize(objective_function, initial_params, args=(y, X, z.values), bounds=bounds, method='L-BFGS-B')

# Estimated parameters
est_params = result.x

# Extract estimated parameters
beta_linear_est = est_params[:k]
beta_nonlinear_est = est_params[k:2*k]
gamma_est = est_params[-2]
c_est = est_params[-1]

print("Estimated Linear Coefficients:", beta_linear_est)
print("Estimated Non-Linear Coefficients:", beta_nonlinear_est)
print("Estimated gamma:", gamma_est)
print("Estimated c:", c_est)






# Compute transition function values
G_est = logistic_transition(z.values, gamma_est, c_est)

# Compute fitted values
y_fitted = X.dot(beta_linear_est) + (X.iloc[:, 1:] * G_est[:, np.newaxis]).sum(axis=1) * beta_nonlinear_est[1:].sum() + beta_nonlinear_est[0] * G_est

# Compute residuals
residuals = y - y_fitted



# Plot residuals
plt.figure(figsize=(10, 5))
plt.plot(residuals, label='STR Model Residuals')
plt.legend()
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()



# Durbin-Watson statistic
dw_stat_str = durbin_watson(residuals)
print(f'Durbin-Watson statistic (STR Model): {dw_stat_str}')

# Ljung-Box test
lb_test_str = acorr_ljungbox(residuals, lags=[12], return_df=True)
print(lb_test_str)

# Breusch-Pagan test
bp_test_str = het_breuschpagan(residuals, X)
bp_test_str_results = dict(zip(['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'], bp_test_str))
print(bp_test_str_results)


#Check for Normality of Residuals
from scipy.stats import jarque_bera

# Perform the Jarque-Bera test
jb_stat, jb_pvalue = jarque_bera(residuals)
print(f'Jarque-Bera test statistic: {jb_stat}')
print(f'p-value: {jb_pvalue}')


import scipy.stats as stats

# Q-Q plot of residuals
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()



import numpy as np

# Extract the inverse Hessian matrix from the optimization result
hessian_inv = result.hess_inv.todense()

# Compute standard errors
standard_errors = np.sqrt(np.diag(hessian_inv))

# Parameter names
param_names = ['const_linear', 'D_ln_CPI_linear', 'D_ln_INDPRO_linear',
               'const_nonlinear', 'D_ln_CPI_nonlinear', 'D_ln_INDPRO_nonlinear',
               'gamma', 'c']

# Create a DataFrame with estimates and standard errors
import pandas as pd

estimates = pd.DataFrame({
    'Parameter': param_names,
    'Estimate': est_params,
    'Std. Error': standard_errors
})

# Calculate t-statistics
estimates['t-stat'] = estimates['Estimate'] / estimates['Std. Error']

# Calculate p-values (two-tailed)
from scipy.stats import t

degrees_of_freedom = len(y) - len(est_params)
estimates['p-value'] = 2 * (1 - t.cdf(np.abs(estimates['t-stat']), df=degrees_of_freedom))

print(estimates)




window_size = 30  # Adjust the window size as needed
rolling_params = []

for start in range(len(y) - window_size + 1):
    end = start + window_size
    y_window = y.iloc[start:end].reset_index(drop=True)
    X_window = X.iloc[start:end].reset_index(drop=True)
    z_window = z[start:end]

    # Re-estimate the model
    result_window = minimize(
        objective_function, initial_params, args=(y_window, X_window, z_window),
        bounds=bounds, method='L-BFGS-B', options={'disp': False}
    )

    if result_window.success:
        est_params_window = result_window.x
        rolling_params.append(est_params_window)
    else:
        # Handle optimization failure
        rolling_params.append([np.nan] * len(initial_params))

# Convert to DataFrame
rolling_params = pd.DataFrame(rolling_params, columns=param_names)
rolling_params['Date'] = y.index[window_size - 1:]

# Plot rolling estimates
plt.figure(figsize=(12, 8))
for param in param_names:
    plt.plot(rolling_params['Date'], rolling_params[param], label=param)

plt.xlabel('Date')
plt.ylabel('Parameter Estimates')
plt.title('Rolling Parameter Estimates')
plt.legend()
plt.show()






# Plot the estimated transition function
plt.figure(figsize=(10, 6))
plt.plot(merged_data.index, G_est, label='Estimated Transition Function G(z_t; gamma, c)')
plt.title('Estimated Transition Function over Time')
plt.xlabel('Date')
plt.ylabel('G(z_t)')
plt.legend()
plt.show()

# Scatter plot of transition function vs. transition variable
plt.figure(figsize=(8, 6))
plt.scatter(z, G_est, alpha=0.5)
plt.title('Transition Function vs. Interest Rate Spread')
plt.xlabel('Interest Rate Spread')
plt.ylabel('G(z_t)')
plt.show()


# Create a summary table
coefficients = pd.DataFrame({
    'Linear Coefficients': beta_linear_est,
    'Non-linear Coefficients': beta_nonlinear_est
}, index=X.columns)

print(coefficients)


# Save the residuals and fitted values
merged_data['STR_Residuals'] = residuals
merged_data['STR_Fitted'] = y_fitted

# Save to CSV
merged_data.to_csv('str_model_results.csv')


