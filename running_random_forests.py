# =============================================================================
# ============= Plotting observables with halo formation times ================
# =============================================================================


# Import some modules
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy as sp
from astropy.cosmology import Planck13 as cosmo


# =============================== Import data =================================
def load_data(with_errors=False):
    print('Loading data and matching...')
    # Import data
    f_core_data = pd.read_csv('/Users/dominicbates/Documents/Python/Halo Formation/data/f_core_cent_plus_sat_200mpc_z015.csv')
    f_half_data = pd.read_csv('/Users/dominicbates/Documents/Python/Halo Formation/data/f_half_cent_plus_sat_200mpc_z015.csv')
    f_vmax_data = pd.read_csv('/Users/dominicbates/Documents/Python/Halo Formation/data/f_vmax_cent_plus_sat_200mpc_z015.csv')
    #obs_data = pd.read_csv('/Users/dominicbates/Documents/Python/Halo Formation/data/some_observables_cent_plus_sat_200mpc_z015.csv')
    obs_data = pd.read_csv('/Users/dominicbates/Documents/Python/Halo Formation/data/some_observables_sfh_cent_plus_sat_200mpc_z015.csv')
    
    # Sorted data
    f_core_data = f_core_data.sort_values('galaxyId')
    f_half_data = f_half_data.sort_values('galaxyId')
    f_vmax_data = f_vmax_data.sort_values('galaxyId')
    obs_data = obs_data.sort_values('galaxyId')
    
    # Rename columns to useful names
    f_core_data = f_core_data.rename(index=str, columns={"minsnap_core_mmean": "f_core_snap", "redshift": "f_core_z"})
    f_half_data = f_half_data.rename(index=str, columns={"minsnap_mmean": "f_half_snap", "redshift": "f_half_z"})
    f_vmax_data = f_vmax_data.rename(index=str, columns={"minsnap_vMax_mmean": "f_vmax_snap", "redshift": "f_vmax_z"})
    
    # Combine to 1 array
    combined_data = pd.merge(obs_data,f_core_data[['galaxyId','f_core_snap','f_core_z']],on='galaxyId')
    combined_data = pd.merge(combined_data,f_half_data[['galaxyId','f_half_snap','f_half_z']],on='galaxyId')
    combined_data = pd.merge(combined_data,f_vmax_data[['galaxyId','f_vmax_snap','f_vmax_z']],on='galaxyId')
    
    # Remove galaxies where stellar mass = 0
    select = combined_data['stellarMass']>0
    combined_data = combined_data.loc[select]
    
    # Keep only centrals (0) or satalites (1)
    select = (combined_data['type']==0) | (obs_data['type']==1)
    combined_data = combined_data.loc[select]
    
    # Return matched arrays (as redshifts)
    f_core_z = np.array(combined_data['f_core_z'])
    f_half_z = np.array(combined_data['f_half_z'])
    f_vmax_z = np.array(combined_data['f_vmax_z'])
                          
    # Create arrays of obersvables (matched)
    gal_sm = np.array(combined_data['stellarMass']) * (10**10) #* 0.7
    gal_type = np.array(combined_data['type'])
    gal_sfr = np.array(combined_data['sfr'])
    gal_ssfr = gal_sfr / gal_sm; 
    gal_mwa = np.array(combined_data['massWeightedAge'])# Units 10**9 yr
    gal_hm = np.array(combined_data['np']) * 8.611e8 #* 0.7
    gal_sh_ratio = gal_sm / gal_hm
    
    # Take log of stellar mass and halo mass
    gal_sm = np.log10(gal_sm)
    gal_hm = np.log10(gal_hm)
    
    # ---------------- Compute ages instead of redshifts
    # (like this because cosmo.age is too slow)
    # Find all unique redshifts and corresponding ages (for each snapshot)
    unique_z = np.unique(np.concatenate((f_core_z,f_half_z,f_vmax_z)))
    unique_age = 13.79629 - cosmo.age(unique_z).value
              
    # Preallocate arrays
    f_core_age = np.zeros(np.size(f_core_z))
    f_half_age = np.zeros(np.size(f_half_z))
    f_vmax_age = np.zeros(np.size(f_vmax_z))
    
    # Loop through array and compute age
    for z in range(np.size(unique_z)):
        # Allocate age for all f_core/half/... at this redshift
        f_core_age[f_core_z == unique_z[z]] = unique_age[z]
        f_half_age[f_half_z == unique_z[z]] = unique_age[z]
        f_vmax_age[f_vmax_z == unique_z[z]] = unique_age[z]
        
    
    # Compute sfh by adding sfh of disk, bulge and icm
    gal_sfh = combined_data.iloc[:,54:74].values#combined_data.iloc[:,14:34].values + combined_data.iloc[:,34:54].values + combined_data.iloc[:,54:74].values

    # Add random errors to data
    if with_errors==True:
        gal_sm = gal_sm + (np.random.randn(np.size(gal_sm))*0.15)
        gal_mwa = gal_mwa + (np.random.randn(np.size(gal_mwa))*1.7)
        
        # Guess errors?
        gal_hm = gal_hm + (np.random.randn(np.size(gal_hm))*0.3)
        gal_sh_ratio = (10**gal_sm) / (10**gal_hm)
        gal_sfr = gal_sfr + (np.random.randn(np.size(gal_hm))*0.3*gal_sfr)
        gal_ssfr = gal_sfr / 10**(gal_sm)
        print('Errors added to observables')

    print('Data loaded')
    
    
    # Return data
    return gal_sm, gal_type, gal_sfr, gal_ssfr, gal_mwa, gal_hm, \
            gal_sh_ratio, gal_sfh, f_core_age, f_half_age, f_vmax_age
# =============================================================================







# ============================== loading data =================================

gal_sm, gal_type, gal_sfr, gal_ssfr, gal_mwa, gal_hm, \
            gal_sh_ratio, gal_sfh, f_core_age, f_half_age, f_vmax_age = load_data()
            
# =============================================================================






# ====================== Setup random forest functions ========================

def run_random_forest(input_data,input_data_df,output_data,print_fi=False):
    # ---------- Split in to training and test data
    from sklearn.model_selection import train_test_split
    train_inp, test_inp, train_out, test_out = train_test_split(input_data, output_data, test_size=0.2, random_state=0)
    
    # ---------- Scaling sample
    # Scale features to normal values (not sure if really important for random forests)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()  
    train_inp = sc.fit_transform(train_inp)  
    test_inp = sc.transform(test_inp)  
    
    # ---------- Training random forest
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=200, random_state=0,n_jobs=-1, min_samples_leaf=20)  
    regressor.fit(train_inp, train_out)  
    test_pred = regressor.predict(test_inp)
    
    # Optionlally print feature importances
    if print_fi==True:
        feature_importances = pd.DataFrame(regressor.feature_importances_,
                                           index = input_data_df.columns,
            columns=['importance']).sort_values('importance',ascending=False)
        print(feature_importances) 
                          
    return test_out, test_pred



def plot_rf_results(test_out, test_pred, p_type=0):
    # Make histogram (p_type = 0), or scatter plot (p_type = 0)
    plt.figure(figsize=(4, 3));
    if p_type == 0:
        plt.hist2d(test_out,test_pred,bins=20,range=[[0,12],[0,12]])
    elif p_type == 1:
        plt.plot(test_out,test_pred,'r.',alpha=0.02,rasterize=True)
    plt.plot([0,12],[0,12],'k-')
    plt.xlabel('$True$')
    plt.ylabel('$Predicted$')
    plt.show() 


def compute_rf_metrics(test_out, test_pred, met_type=1, print_all=False):
    # Import modules
    import scipy as sp
    from sklearn import metrics
    
    # Compute metric for given type (met_type: 1=RMSe, 2=MSe, 3=MAE, 4=SR, 5=PR)
    if met_type == 1:
        metric_out = np.sqrt(metrics.mean_squared_error(test_out, test_pred))
    elif met_type == 2:
        metric_out = metrics.mean_squared_error(test_out, test_pred)
    elif met_type == 3:
        metric_out = metrics.mean_absolute_error(test_out, test_pred)
    elif met_type == 4:
        metric_out = sp.stats.spearmanr(test_out,test_pred)[0]
    elif met_type == 5:
        metric_out = sp.stats.pearsonr(test_out,test_pred)[0]
    
    # Print all parameters
    if print_all==True:
        print('Mean Absolute Error:', metrics.mean_absolute_error(test_out, test_pred))  
        print('Mean Squared Error:', metrics.mean_squared_error(test_out, test_pred))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_out, test_pred)))
        print('Spearman r:', sp.stats.spearmanr(test_out,test_pred)[0])
        print('Pearson r:', sp.stats.pearsonr(test_out,test_pred)[0])      
    
    # Return given metric
    return metric_out
    
    
    
def print_correlation(parameter_1,form_time):
    print('Spearman r:', sp.stats.spearmanr(parameter_1,form_time)[0])
    print('Pearson r:', sp.stats.pearsonr(parameter_1,form_time)[0])
    return



def compute_1d_rf_rmse(parameter_1,form_time):
    
    output_data = np.copy(form_time)

    # Comute with just 1 parameter (zeros are uncorrelated)
    input_data = np.column_stack((parameter_1,np.zeros(np.size(parameter_1))))
    input_data_df = pd.DataFrame(input_data, columns=['shouldnt_print', '0s'])
    test_true,test_pred = run_random_forest(input_data,input_data_df,output_data,False)
    
    # Evaluate performance
    rmse = compute_rf_metrics(test_true, test_pred, 1)
    #mae = compute_rf_metrics(test_true, test_pred, 3)              
    return rmse



def compute_nd_rf_rmse(input_data, input_data_df,form_time):
    
    output_data = np.copy(form_time)
    test_true,test_pred = run_random_forest(input_data,input_data_df,output_data,False)
    
    # Evaluate performance
    rmse = compute_rf_metrics(test_true, test_pred, 1)
    #mae = compute_rf_metrics(test_true, test_pred, 3)              
    return rmse



def compute_running_median(x,y,x_min=False,x_max=False,n_b=100):
    if (x_min==False) or (x_max==False):
        x_min = np.min(x)
        x_max = np.max(x)
        
    # Create bins limits
    bins = np.linspace(x_min,x_max,n_b+1)
    # Find centre of bins
    bin_cents = bins[0:-1] + (bins[1]-bins[0])/2
    # Loop through and compute median
    running_median = np.zeros(np.size(bin_cents))
    # Loop through and compute median / percentile
    for n in range(np.size(bin_cents)):
        bin_y = y[(x>=bins[n])&(x<bins[n+1])]
        running_median[n] = np.median(bin_y)
#        running_perc_u[n] = np.percentile(bin_y,16) # for 1 s.d. etc.
    return bins,bin_cents,running_median

def compute_1d_run_med_rmse(parameter_1,form_time):

    # Split to test and training data
    from sklearn.model_selection import train_test_split
    train_inp, test_inp, train_out, test_out = train_test_split(parameter_1, form_time, test_size=0.2, random_state=0)


    # Find best size of bins
    bin_size = int(np.size(parameter_1**(1/3)))
    if bin_size < 1:
        bin_size = 1
    elif bin_size > 15:
        bin_size = 10
    
    # Compute running median for training data in small bins
    bins, bin_cents, running_median = compute_running_median(train_inp,train_out,n_b=2)
    #print(running_median)    
    
    # Predict output
    test_pred = np.zeros(np.size(test_inp))
    # Loop through all bins of p1 (in test data)
    for n in range(np.size(bin_cents)):
        # Select objects in this bin of p1
        sel = (test_inp >= bins[n]) * (test_inp < bins[n+1])
        test_pred[sel] = running_median[n]
        
    #print(test_pred)
        
    # Evaluate performance
    rmse = compute_rf_metrics(test_out, test_pred, 1)
    #mae = compute_rf_metrics(test_true, test_pred, 3)              
    return rmse

        


def compute_1d_polyfit_rmse(parameter_1,form_time):

    def polyfit_prediction(train_inp, train_out, test_inp, deg=1):
        # Fit polynomial
        z = np.polyfit(train_inp,train_out,deg)
        
        # Predict output
        if deg==1:
            test_pred = test_inp*z[0] + z[1]
        elif deg==2:
            test_pred = (test_inp**2)*z[0] + test_inp*z[1] + z[2]
        elif deg==3:
            test_pred = (test_inp**3)*z[0] + (test_inp**2)*z[1] + test_inp*z[2] + z[3]
        
        return test_pred
    
    # Split to test and training data
    from sklearn.model_selection import train_test_split
    train_inp, test_inp, train_out, test_out = train_test_split(parameter_1, form_time, test_size=0.2, random_state=0)

    # Compute prediction from polynomial fit
    test_pred = polyfit_prediction(train_inp, train_out, test_inp, 1)
        
    # Evaluate performance
    rmse = compute_rf_metrics(test_out, test_pred, 1)
    #mae = compute_rf_metrics(test_true, test_pred, 3)              
    return rmse

        

# =============================================================================







# ================ Compute RMSe for 1 param in halo mass bins==================


# Trying with just sm, ssfr, mwa
just_sm_mwa_ssfr = np.column_stack((gal_sm, gal_ssfr, gal_mwa))
just_sm_mwa_ssfr_df = pd.DataFrame(just_sm_mwa_ssfr, columns=['sm', 'ssfr', 'mwa'])

# Trying with all parameters minus halo mass and sfh
all_min_h_sfh = np.column_stack((gal_sm, gal_sfr, gal_ssfr, gal_mwa, gal_sh_ratio))
all_min_h_sfh_df = pd.DataFrame(all_min_h_sfh, columns=['sm', 'sfr', 'ssfr', 'mwa', 'sh_ratio'])

# Trying with all parameters minus sfh
all_min_sfh = np.column_stack((gal_sm, gal_sfr, gal_ssfr, gal_mwa, gal_sh_ratio, gal_hm))
all_min_sfh_df = pd.DataFrame(all_min_sfh, columns=['sm', 'sfr', 'ssfr', 'mwa', 'sh_ratio', 'hm'])

# Trying with all parameters minus halo mass
all_min_h = np.concatenate([all_min_h_sfh,gal_sfh],axis=1)
all_min_h_df = pd.DataFrame(all_min_h, columns=['sm', 'sfr', 'ssfr', 'mwa', 'sh_ratio','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])

# Trying with all parameters
all_par = np.concatenate([all_min_sfh,gal_sfh],axis=1)
all_par_df = pd.DataFrame(all_par, columns=['sm', 'sfr', 'ssfr', 'mwa', 'sh_ratio','hm','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])

# Trying with just SFH
just_sfh = np.copy(gal_sfh)
just_sfh_df = pd.DataFrame(just_sfh, columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])



# Define bins of halo mass
hm_bins = np.linspace(11.5,13.5,9)
hm_bins = np.linspace(11.5,13.5,11)
hm_bins_cent = (hm_bins[0:-1] + hm_bins[1:])/2

# Create blank arrays to store RMS errors
# -----
rmse_fh_sm = np.zeros(hm_bins.size-1) # for f_half
rmse_fh_mwa = np.zeros(hm_bins.size-1)
rmse_fh_sfr = np.zeros(hm_bins.size-1)
rmse_fh_ssfr = np.zeros(hm_bins.size-1)
rmse_fh_sh = np.zeros(hm_bins.size-1)
rmse_fh_hm = np.zeros(hm_bins.size-1)
rmse_fh_all = np.zeros(hm_bins.size-1)
rmse_fh_all_sfh = np.zeros(hm_bins.size-1)
rmse_fh_all_sfh_hm = np.zeros(hm_bins.size-1)
rmse_fh_sm_mwa_ssfr = np.zeros(hm_bins.size-1)
rmse_fh_just_sfh = np.zeros(hm_bins.size-1)
# -----
rmse_fv_sm = np.zeros(hm_bins.size-1) # for f_vmax
rmse_fv_mwa = np.zeros(hm_bins.size-1)
rmse_fv_sfr = np.zeros(hm_bins.size-1)
rmse_fv_ssfr = np.zeros(hm_bins.size-1)
rmse_fv_sh = np.zeros(hm_bins.size-1)
rmse_fv_hm = np.zeros(hm_bins.size-1)
rmse_fv_all = np.zeros(hm_bins.size-1)
rmse_fv_all_sfh = np.zeros(hm_bins.size-1)
rmse_fv_all_sfh_hm = np.zeros(hm_bins.size-1)
rmse_fv_sm_mwa_ssfr = np.zeros(hm_bins.size-1)
rmse_fv_just_sfh = np.zeros(hm_bins.size-1)
# -----
rmse_fc_sm = np.zeros(hm_bins.size-1) # for f_core
rmse_fc_mwa = np.zeros(hm_bins.size-1)
rmse_fc_sfr = np.zeros(hm_bins.size-1)
rmse_fc_ssfr = np.zeros(hm_bins.size-1)
rmse_fc_sh = np.zeros(hm_bins.size-1)
rmse_fc_hm = np.zeros(hm_bins.size-1)
rmse_fc_all = np.zeros(hm_bins.size-1)
rmse_fc_all_sfh = np.zeros(hm_bins.size-1)
rmse_fc_all_sfh_hm = np.zeros(hm_bins.size-1)
rmse_fc_sm_mwa_ssfr = np.zeros(hm_bins.size-1)
rmse_fc_just_sfh = np.zeros(hm_bins.size-1)

# Loop though halo mass bins
for h in range(hm_bins.size-1):
    # Select galaxies in this halo mass bin
    sel = (gal_hm >= hm_bins[h]) * (gal_hm < hm_bins[h+1])
    
    # Compute rms error for each parameter
    # -----
    rmse_fh_sm[h] = compute_1d_polyfit_rmse(gal_sm[sel],f_half_age[sel])
    rmse_fh_mwa[h] = compute_1d_polyfit_rmse(gal_mwa[sel],f_half_age[sel])
    rmse_fh_sfr[h] = compute_1d_polyfit_rmse(gal_sfr[sel],f_half_age[sel])
    rmse_fh_ssfr[h] = compute_1d_polyfit_rmse(gal_ssfr[sel],f_half_age[sel])
    rmse_fh_sh[h] = compute_1d_polyfit_rmse(gal_sh_ratio[sel],f_half_age[sel])
    rmse_fh_hm[h] = compute_1d_polyfit_rmse(gal_hm[sel],f_half_age[sel])
    # -----
    rmse_fv_sm[h] = compute_1d_polyfit_rmse(gal_sm[sel],f_vmax_age[sel])
    rmse_fv_mwa[h] = compute_1d_polyfit_rmse(gal_mwa[sel],f_vmax_age[sel])
    rmse_fv_sfr[h] = compute_1d_polyfit_rmse(gal_sfr[sel],f_vmax_age[sel])
    rmse_fv_ssfr[h] = compute_1d_polyfit_rmse(gal_ssfr[sel],f_vmax_age[sel])
    rmse_fv_sh[h] = compute_1d_polyfit_rmse(gal_sh_ratio[sel],f_vmax_age[sel])
    rmse_fv_hm[h] = compute_1d_polyfit_rmse(gal_hm[sel],f_vmax_age[sel])
    # -----
    rmse_fc_sm[h] = compute_1d_polyfit_rmse(gal_sm[sel],f_core_age[sel])
    rmse_fc_mwa[h] = compute_1d_polyfit_rmse(gal_mwa[sel],f_core_age[sel])
    rmse_fc_sfr[h] = compute_1d_polyfit_rmse(gal_sfr[sel],f_core_age[sel])
    rmse_fc_ssfr[h] = compute_1d_polyfit_rmse(gal_ssfr[sel],f_core_age[sel])
    rmse_fc_sh[h] = compute_1d_polyfit_rmse(gal_sh_ratio[sel],f_core_age[sel])
    rmse_fc_hm[h] = compute_1d_polyfit_rmse(gal_hm[sel],f_core_age[sel])
    
    
    # Compute rms error for several parameters
    # -----
    rmse_fh_all[h] = compute_nd_rf_rmse(all_min_h_sfh[sel], all_min_h_sfh_df[sel], f_half_age[sel])
    rmse_fh_all_sfh[h] = compute_nd_rf_rmse(all_min_h[sel], all_min_h_df[sel], f_half_age[sel])
    rmse_fh_all_sfh_hm[h] = compute_nd_rf_rmse(all_par[sel], all_par_df[sel], f_half_age[sel])
    rmse_fh_sm_mwa_ssfr[h] = compute_nd_rf_rmse(just_sm_mwa_ssfr[sel], just_sm_mwa_ssfr_df[sel], f_half_age[sel])
    rmse_fh_just_sfh[h] = compute_nd_rf_rmse(just_sfh[sel], just_sfh_df[sel], f_half_age[sel])
    # -----
    rmse_fv_all[h] = compute_nd_rf_rmse(all_min_h_sfh[sel], all_min_h_sfh_df[sel], f_vmax_age[sel])
    rmse_fv_all_sfh[h] = compute_nd_rf_rmse(all_min_h[sel], all_min_h_df[sel], f_vmax_age[sel])
    rmse_fv_all_sfh_hm[h] = compute_nd_rf_rmse(all_par[sel], all_par_df[sel], f_vmax_age[sel])
    rmse_fv_sm_mwa_ssfr[h] = compute_nd_rf_rmse(just_sm_mwa_ssfr[sel], just_sm_mwa_ssfr_df[sel], f_vmax_age[sel])
    rmse_fv_just_sfh[h] = compute_nd_rf_rmse(just_sfh[sel], just_sfh_df[sel], f_vmax_age[sel])
    # -----
    rmse_fc_all[h] = compute_nd_rf_rmse(all_min_h_sfh[sel], all_min_h_sfh_df[sel], f_core_age[sel])
    rmse_fc_all_sfh[h] = compute_nd_rf_rmse(all_min_h[sel], all_min_h_df[sel], f_core_age[sel])
    rmse_fc_all_sfh_hm[h] = compute_nd_rf_rmse(all_par[sel], all_par_df[sel], f_core_age[sel])  
    rmse_fc_sm_mwa_ssfr[h] = compute_nd_rf_rmse(just_sm_mwa_ssfr[sel], just_sm_mwa_ssfr_df[sel], f_core_age[sel])
    rmse_fc_just_sfh[h] = compute_nd_rf_rmse(just_sfh[sel], just_sfh_df[sel], f_core_age[sel])
        
    print(str(((h+1)/(hm_bins.size-1))*100)+'% complete')







# =================== Plotting results from single paramters ==================

fig = plt.figure(figsize=(4.3,9))

# Plotting results for f_half
plt.subplot(3, 1, 1)
# Halo mass
plt.plot(hm_bins_cent, rmse_fh_hm,'k-',linewidth=2,linestyle='dashed',alpha=0.5)
# Other parameters
plt.plot(hm_bins_cent, rmse_fh_sm,'b-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fh_mwa,'g-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fh_sfr,'k-',color='orange',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fh_ssfr,'r-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fh_sh,'k-',color='brown',linewidth=2,alpha=0.5)
plt.ylim([0.8,2.3])
plt.xlim([11.5,13.5])
plt.ylabel('$\sigma_{RMS}$ $(GYr)$')
plt.xlabel('$M_{h}$')
plt.text(13.4, 2.18, '$M_{\star}$', color='b',horizontalalignment='right',fontsize=12)
plt.text(13.4, 2.04, '$MWA$', color='g',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.9, '$SFR$', color='orange',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.76, '$sSFR$', color='r',horizontalalignment='right',fontsize=12)
plt.text(11.58, 2.13, '$f_{half}$', color='k',horizontalalignment='left',fontsize=16)

# Plotting results for f_vmax
plt.subplot(3, 1, 2)
# Halo mass
plt.plot(hm_bins_cent, rmse_fv_hm,'k-',linewidth=2,linestyle='dashed',alpha=0.5)
# Other parameters
plt.plot(hm_bins_cent, rmse_fv_sm,'b-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fv_mwa,'g-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fv_sfr,'k-',color='orange',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fv_ssfr,'r-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fv_sh,'k-',color='brown',linewidth=2,alpha=0.5)
plt.ylim([2,3.2])
plt.xlim([11.5,13.5])
plt.ylabel('$\sigma_{RMS}$ $(GYr)$')
plt.xlabel('$M_{h}$')
plt.text(13.4, 3.1, '$M_{\star}$', color='b',horizontalalignment='right',fontsize=12)
plt.text(13.4, 2.98, '$MWA$', color='g',horizontalalignment='right',fontsize=12)
plt.text(13.4, 2.86, '$SFR$', color='orange',horizontalalignment='right',fontsize=12)
plt.text(13.4, 2.74, '$sSFR$', color='r',horizontalalignment='right',fontsize=12)
plt.text(11.58, 3.06, '$f_{vmax}$', color='k',horizontalalignment='left',fontsize=16)


# Plotting results for f_core
plt.subplot(3, 1, 3)
# Halo mass
plt.plot(hm_bins_cent, rmse_fc_hm,'k-',linewidth=2,linestyle='dashed',alpha=0.5)
# Other parameters
plt.plot(hm_bins_cent, rmse_fc_sm,'b-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fc_mwa,'g-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fc_sfr,'k-',color='orange',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fc_ssfr,'r-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fc_sh,'k-',color='brown',linewidth=2,alpha=0.5)
plt.ylim([0,2.3])
plt.xlim([11.5,13.5])
plt.ylabel('$\sigma_{RMS}$ $(GYr)$')
plt.xlabel('$M_{h}$')
plt.text(13.4, 2.1, '$M_{\star}$', color='b',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.86, '$MWA$', color='g',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.62, '$SFR$', color='orange',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.38, '$sSFR$', color='r',horizontalalignment='right',fontsize=12)
plt.text(11.58, 2.05, '$f_{core}$', color='k',horizontalalignment='left',fontsize=16)

# Set tight layout and save
fig.tight_layout()
#plt.savefig('thesis_plot_ch4_1_parameter.pdf',bbox_inches='tight')



# =================== Plotting results from random forests ====================

fig = plt.figure(figsize=(4.3,9))

# Plotting results for f_half
plt.subplot(3, 1, 1)
# Halo mass
plt.plot(hm_bins_cent, rmse_fh_hm,'k-',linewidth=2,linestyle='dashed',alpha=0.5)
# Other parameters
plt.plot(hm_bins_cent, rmse_fh_all,'k-',color='deeppink',linewidth=2,alpha=0.5)
#plt.plot(hm_bins_cent, rmse_fh_all_sfh,'c-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fh_all_sfh_hm,'k-',color='navy',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fh_sm_mwa_ssfr,'c-',linewidth=2,alpha=0.5)
plt.ylim([0.8,2.3])
plt.xlim([11.5,13.5])
plt.ylabel('$\sigma_{RMS}$ $(GYr)$')
plt.xlabel('$M_{h}$')
plt.text(13.4, 2.14, '$M_{\star} + MWA+ sSFR$', color='c',horizontalalignment='right',fontsize=12)
plt.text(13.4, 2.00, '$... + s/h$', color='deeppink',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.86, '$... + SFH$', color='navy',horizontalalignment='right',fontsize=12)
plt.text(11.58, 2.1, '$f_{half}$', color='k',horizontalalignment='left',fontsize=16)


# Plotting results for f_vmax
plt.subplot(3, 1, 2)
# Halo mass
plt.plot(hm_bins_cent, rmse_fv_hm,'k-',linewidth=2,linestyle='dashed',alpha=0.5)
# Other parameters
plt.plot(hm_bins_cent, rmse_fv_all,'k-',color='deeppink',linewidth=2,alpha=0.5)
#plt.plot(hm_bins_cent, rmse_fv_all_sfh,'c-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fv_all_sfh_hm,'k-',color='navy',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fv_sm_mwa_ssfr,'c-',linewidth=2,alpha=0.5)
plt.ylim([2,3.2])
plt.xlim([11.5,13.5])
plt.ylabel('$\sigma_{RMS}$ $(GYr)$')
plt.xlabel('$M_{h}$')
plt.text(13.4, 3.1, '$M_{\star} + MWA+ sSFR$', color='c',horizontalalignment='right',fontsize=12)
plt.text(13.4, 2.98, '$... + s/h$', color='deeppink',horizontalalignment='right',fontsize=12)
plt.text(13.4, 2.86, '$... + SFH$', color='navy',horizontalalignment='right',fontsize=12)
plt.text(11.58, 3.06, '$f_{vmax}$', color='k',horizontalalignment='left',fontsize=16)


# Plotting results for f_core
plt.subplot(3, 1, 3)
# Halo mass
plt.plot(hm_bins_cent, rmse_fc_hm,'k-',linewidth=2,linestyle='dashed',alpha=0.5)
# Other parameters
plt.plot(hm_bins_cent, rmse_fc_all,'k-',color='deeppink',linewidth=2,alpha=0.5)
#plt.plot(hm_bins_cent, rmse_fc_all_sfh,'c-',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fc_all_sfh_hm,'k-',color='navy',linewidth=2,alpha=0.5)
plt.plot(hm_bins_cent, rmse_fc_sm_mwa_ssfr,'c-',linewidth=2,alpha=0.5)
plt.ylim([0,2.3])
plt.xlim([11.5,13.5])
plt.ylabel('$\sigma_{RMS}$ $(GYr)$')
plt.xlabel('$M_{h}$')
plt.text(13.4, 2.1, '$M_{\star} + MWA+ sSFR$', color='c',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.86, '$... + s/h$', color='deeppink',horizontalalignment='right',fontsize=12)
plt.text(13.4, 1.62, '$... + SFH$', color='navy',horizontalalignment='right',fontsize=12)
plt.text(11.58, 2.05, '$f_{core}$', color='k',horizontalalignment='left',fontsize=16)

# Set tight layout and save
fig.tight_layout()
#plt.savefig('thesis_plot_ch4_random_forests.pdf',bbox_inches='tight')