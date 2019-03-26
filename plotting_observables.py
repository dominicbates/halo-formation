# =============================================================================
# ============= Plotting observables with halo formation times ================
# =============================================================================


# Import some modules
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy as sp
from astropy.cosmology import Planck13 as cosmo




# =============================== Import data =================================

def load_data():
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

    print('Data loaded')
    # Return data
    return gal_sm, gal_type, gal_sfr, gal_ssfr, gal_mwa, gal_hm, \
            gal_sh_ratio, gal_sfh, f_core_age, f_half_age, f_vmax_age
# =============================================================================






# ======================== Some plotting functions ============================

def compute_running_median_sd(x,y,x_min,x_max,b):
 # Create bins limits
    bins = np.linspace(x_min,x_max,b+1)
    # Find centre of bins
    bin_cents = bins[0:-1] + (bins[1]-bins[0])/2
    # Loop through and compute median
    running_median = np.zeros(np.size(bin_cents))
    running_perc_u = np.zeros(np.size(bin_cents))
    running_perc_l = np.zeros(np.size(bin_cents))
    # Loop through and compute median / percentile
    for n in range(np.size(bin_cents)):
        bin_y = y[(x>=bins[n])&(x<bins[n+1])]
        running_median[n] = np.median(bin_y)
        running_perc_u[n] = np.percentile(bin_y,16)
        running_perc_l[n] = np.percentile(bin_y,84)
    return bin_cents,running_median,running_perc_l,running_perc_u
    
def plot_x_vs_y(x,y,x_lims,y_lims,b,xlabel,ylabel,col,subplot=False,yax=True,xax=True):
    x = globals()[x]
    y = globals()[y]
    # Create new figure unless subplot = True
    if subplot==False:
        plt.figure(figsize=(4,3))
    plt.plot(x,y,'.',alpha=0.02,markersize=2,color=col, rasterized=True);plt.xlim(x_lims);plt.ylim(y_lims)
    #plt.hist2d(x,y,bins=b,range=[x_lims,y_lims])
    # Plot axis
    if xax==True:
        plt.xlabel(xlabel,fontsize=14)
    if yax==True:
        plt.ylabel(ylabel,fontsize=14)
    
    # Plot median
    bin_cents, med, up, low = compute_running_median_sd(x,y,x_lims[0],x_lims[-1],b)
    plt.plot(bin_cents,med,'r-',lw=2.5,alpha=.8)
    plt.plot(bin_cents,low,'r--',lw=2.5,alpha=.8)
    plt.plot(bin_cents,up,'r--',lw=2.5,alpha=.8)
    return

def plot_x_vs_y_hm_bin(xn,yn,x_lims,y_lims,b,xlabel,ylabel,col,hm,hm_lims):
    x = np.copy(globals()[xn])
    y = np.copy(globals()[yn])
    select = (hm > hm_lims[0]) & (hm < hm_lims[1])
    x=x[select]; y=y[select]
    plt.figure(figsize=(4,3))
    plt.plot(x,y,'.',alpha=0.02,markersize=2,color=col, rasterized=True);plt.xlim(x_lims);plt.ylim(y_lims)
    #plt.hist2d(x,y,bins=b,range=[x_lims,y_lims])
    plt.xlabel(xlabel,fontsize=14)
    plt.ylabel(ylabel,fontsize=14)
    
    # Plot median
    bin_cents, med, up, low = compute_running_median_sd(x,y,np.percentile(x,2),np.percentile(x,98),5)
    plt.plot(bin_cents,med,'r-',lw=2.5,alpha=.8)
    plt.plot(bin_cents,low,'r--',lw=2.5,alpha=.8)
    plt.plot(bin_cents,up,'r--',lw=2.5,alpha=.8)
    return

def compute_hist_percentile(bin_cent,counts,percentile_n):
    # Import modules
    import numpy as np
    percentile_n = percentile_n/100
    # Compute sum of counts
    total_count = np.sum(counts)
    # Assign cumiliative counts
    cum_count = np.zeros(np.size(counts))
    for n in range(np.size(cum_count)):
        if n > 0:
            cum_count[n] = cum_count[n-1] + counts[n]
        else:
            cum_count[n] = counts[n]
    cum_count = cum_count / total_count
    # Work out bin location either side of Nth percentile
    pos_max = np.argmax(cum_count>percentile_n)
    pos_min = np.size(cum_count) - 1 - np.argmax(np.flip(cum_count,axis=0)<0.5)
    # Compute interpolated mean between these two bins
    d_high = cum_count[pos_max] - percentile_n
    d_low = percentile_n - cum_count[pos_min]
    percentile_val = (bin_cent[pos_max] - bin_cent[pos_min]) * \
                     (d_low/(d_low + d_high)) + bin_cent[pos_min]
    # Return the value at Nth percentile
    return percentile_val


# =============================================================================





# ============================== loading data =================================

gal_sm, gal_type, gal_sfr, gal_ssfr, gal_mwa, gal_hm, \
            gal_sh_ratio, gal_sfh, f_core_age, f_half_age, f_vmax_age = load_data()
            
# =============================================================================










# ======================= Plotting some observables ===========================


# ----- halo mass vs formation times (same figure) - looks best this way round
fig = plt.figure(figsize=(4.3,9))
# plot fhalf
plt.subplot(3, 1, 1)
plot_x_vs_y('gal_hm','f_half_age',[11.5,14.5],[2,14],25,'$M_{halo}$ $(M_{\odot})$','$f_{half}$ $(Gyr)$','k',True)
# plot fvmax
plt.subplot(3, 1, 2)
plot_x_vs_y('gal_hm','f_vmax_age',[11.5,14.5],[2,14],25,'$M_{halo}$ $(M_{\odot})$','$f_{vmax}$ $(Gyr)$','k',True)
# plot fcore
plt.subplot(3, 1, 3)
plot_x_vs_y('gal_hm','f_core_age',[11.5,14.5],[2,14],25,'$M_{halo}$ $(M_{\odot})$','$f_{core}$ $(Gyr)$','k',True)
# Force tight layout and save
fig.tight_layout()
#plt.savefig('thesis_plot_ch4_hm_vs_times.pdf',bbox_inches='tight')


# ----- formation time vs all parameters (same figure)
fig = plt.figure(figsize=(10,14))
# formation times vs stellar mass
plt.subplot(5, 3, 1); plot_x_vs_y('f_half_age','gal_sm',[2,13],[9,12],25,'$f_{half}$ $(Gyr)$','$M_{\star}$ $(M_{\odot})$','b',True,True,False)
plt.subplot(5, 3, 2); plot_x_vs_y('f_vmax_age','gal_sm',[2,13],[9,12],25,'$f_{vmax}$ $(Gyr)$','$M_{\star}$ $(M_{\odot})$','b',True,False,False)
plt.subplot(5, 3, 3); plot_x_vs_y('f_core_age','gal_sm',[2,13],[9,12],25,'$f_{core}$ $(Gyr)$','$M_{\star}$ $(M_{\odot})$','b',True,False,False)

# formation times vs mass weighted age
plt.subplot(5, 3, 4); plot_x_vs_y('f_half_age','gal_mwa',[2,13],[2,10],25,'$f_{half}$ $(Gyr)$','$MWA$ $(GYr)$','g',True,True,False)
plt.subplot(5, 3, 5); plot_x_vs_y('f_vmax_age','gal_mwa',[2,13],[2,10],25,'$f_{vmax}$ $(Gyr)$','$MWA$ $(GYr)$','g',True,False,False)
plt.subplot(5, 3, 6); plot_x_vs_y('f_core_age','gal_mwa',[2,13],[2,10],25,'$f_{core}$ $(Gyr)$','$MWA$ $(GYr)$','g',True,False,False)

# formation times vs sfr
plt.subplot(5, 3, 7); plot_x_vs_y('f_half_age','gal_sfr',[2,13],[0,8],25,'$f_{half}$ $(Gyr)$','$SFR$ $(M_{\odot} Yr^{-1})$','orange',True,True,False)
plt.subplot(5, 3, 8); plot_x_vs_y('f_vmax_age','gal_sfr',[2,13],[0,8],25,'$f_{vmax}$ $(Gyr)$','$SRF$ $(M_{\odot} Yr^{-1})$','orange',True,False,False)
plt.subplot(5, 3, 9); plot_x_vs_y('f_core_age','gal_sfr',[2,13],[0,8],25,'$f_{core}$ $(Gyr)$','$SRF$ $(M_{\odot} Yr^{-1})$','orange',True,False,False)

# formation times vs ssfr
plt.subplot(5, 3, 10); plot_x_vs_y('f_half_age','gal_ssfr',[2,13],[0,0.8e-9],25,'$f_{half}$ $(Gyr)$','$sSRF$  $(Yr^{-1})$','r',True,True,False)
plt.subplot(5, 3, 11); plot_x_vs_y('f_vmax_age','gal_ssfr',[2,13],[0,0.8e-9],25,'$f_{vmax}$ $(Gyr)$','$sSRF$  $(Yr^{-1})$','r',True,False,False)
plt.subplot(5, 3, 12); plot_x_vs_y('f_core_age','gal_ssfr',[2,13],[0,0.8e-9],25,'$f_{core}$ $(Gyr)$','$sSRF$  $(Yr^{-1})$','r',True,False,False)

# formation times vs stellar/halo mass ratio
plt.subplot(5, 3, 13); plot_x_vs_y('f_half_age','gal_sh_ratio',[2,13],[0,0.05],25,'$f_{half}$ $(Gyr)$','$s/h$ $ratio$','m',True,True,True)
plt.subplot(5, 3, 14); plot_x_vs_y('f_vmax_age','gal_sh_ratio',[2,13],[0,0.05],25,'$f_{vmax}$ $(Gyr)$','$s/h$ $ratio$','m',True,False,True)
plt.subplot(5, 3, 15); plot_x_vs_y('f_core_age','gal_sh_ratio',[2,13],[0,0.05],25,'$f_{core}$ $(Gyr)$','$s/h$ $ratio$','m',True,False,True)

fig.tight_layout()
#plt.savefig('thesis_plot_ch4_times_vs_all_par.pdf',bbox_inches='tight')




fig = plt.figure(figsize=(4.3,10))
# plot fhalf
plt.subplot(4, 1, 1)
plot_x_vs_y('gal_hm','gal_sm',[11.5,14.5],[9,12],25,'$M_{halo}$ $(M_{\odot})$','$M_{\star}$ $(M_{\odot})$','b',True,True,False)# plot fvmax
plt.subplot(4, 1, 2)
plot_x_vs_y('gal_hm','gal_mwa',[11.5,14.5],[2,12],25,'$M_{halo}$ $(M_{\odot})$','$MWA$ $(GYr)$','g',True,True,False)
plt.subplot(4, 1, 3)
plot_x_vs_y('gal_hm','gal_sfr',[11.5,14.5],[0,8],25,'$M_{halo}$ $(M_{\odot})$','$SFR$ $(M_{\odot} Yr^{-1})$','orange',True,True,False)
plt.subplot(4, 1, 4)
plot_x_vs_y('gal_hm','gal_ssfr',[11.5,14.5],[0,0.8e-9],25,'$M_{halo}$ $(M_{\odot})$','$sSRF$  $(Yr^{-1})$','r',True,True,True)

# Force tight layout and save
fig.tight_layout()
#plt.savefig('thesis_plot_ch4_hm_vs_all.pdf',bbox_inches='tight')




# =============================================================================






# ====================== Setup random forst functions =========================

def run_random_forest(input_data,input_data_df,output_data,printyn=True):
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
    regressor = RandomForestRegressor(n_estimators=50, random_state=0,n_jobs=-1)  
    regressor.fit(train_inp, train_out)  
    test_pred = regressor.predict(test_inp)
    
    # ---------- Evaluate performance
    
    # Print is printyn is true
    from sklearn import metrics
    if printyn == True:
        print('Mean Absolute Error:', metrics.mean_absolute_error(test_out, test_pred))  
        #print('Mean Squared Error:', metrics.mean_squared_error(test_out, test_pred))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_out, test_pred)))
        #print('Feature importances: ',regressor.feature_importances_)
        print('Spearman r:', sp.stats.spearmanr(test_out,test_pred)[0])
        print('Pearson r:', sp.stats.pearsonr(test_out,test_pred)[0])
        feature_importances = pd.DataFrame(regressor.feature_importances_,
                                           index = input_data_df.columns,
                                            columns=['importance']).sort_values('importance',ascending=False)
        
        print(feature_importances)
        plt.figure(figsize=(4, 3));
        #plt.plot(test_out,test_pred,'r.',alpha=0.02)
        plt.hist2d(test_out,test_pred,bins=20,range=[[0,12],[0,12]])
        plt.plot([0,12],[0,12],'k-')
        plt.xlabel('$True$')
        plt.ylabel('$Predicted$')
        plt.show()                       
    return test_out, test_pred


def print_correlation(parameter_1,form_time):
    print('Spearman r:', sp.stats.spearmanr(parameter_1,form_time)[0])
    print('Pearson r:', sp.stats.pearsonr(parameter_1,form_time)[0])
    return


def compute_1d_random_forest_stats(parameter_1,form_time):
    
    output_data = np.copy(form_time)

    # Trying with just 1 parameters (most correlated)
    input_data = np.column_stack((parameter_1,np.zeros(np.size(parameter_1))))
    input_data_df = pd.DataFrame(input_data, columns=['shouldnt_print', '0s'])
    test_true,test_pred = run_random_forest(input_data,input_data_df,output_data,False)
    
    # Evaluate performance
    from sklearn import metrics
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_true, test_pred)))
    print('Mean Absolute Squared Error:', np.sqrt(metrics.mean_squared_error(test_true, test_pred)))
#    print('Spearman r:', sp.stats.spearmanr(test_true, test_pred)[0])
#    print('Pearson r:', sp.stats.pearsonr(test_true, test_pred)[0])
    return

# =============================================================================








# ============ Running random forests for individual paramters ================

# Set formation time to f_half
output_data = np.copy(f_half_age)

# Copmute
compute_1d_random_forest_stats(gal_sm,output_data)
compute_1d_random_forest_stats(gal_mwa,output_data)
compute_1d_random_forest_stats(gal_sfr,output_data)
compute_1d_random_forest_stats(gal_ssfr,output_data)
compute_1d_random_forest_stats(gal_sh_ratio,output_data)
compute_1d_random_forest_stats(gal_hm,output_data)

# Set formation time to f_vmax
output_data = np.copy(f_vmax_age)

# Copmute
compute_1d_random_forest_stats(gal_sm,output_data)
compute_1d_random_forest_stats(gal_mwa,output_data)
compute_1d_random_forest_stats(gal_sfr,output_data)
compute_1d_random_forest_stats(gal_ssfr,output_data)
compute_1d_random_forest_stats(gal_sh_ratio,output_data)
compute_1d_random_forest_stats(gal_hm,output_data)

# Set formation time to f_vmax
output_data = np.copy(f_core_age)

# Copmute
compute_1d_random_forest_stats(gal_sm,output_data)
compute_1d_random_forest_stats(gal_mwa,output_data)
compute_1d_random_forest_stats(gal_sfr,output_data)
compute_1d_random_forest_stats(gal_ssfr,output_data)
compute_1d_random_forest_stats(gal_sh_ratio,output_data)
compute_1d_random_forest_stats(gal_hm,output_data)

# =============================================================================



















# ---------- Running random forests for different parameters 

# ----- For f_half
output_data = np.copy(f_half_age)

# Trying with just 1 parameters (most correlated)
input_data = np.column_stack((gal_sh_ratio ,np.zeros(np.size(gal_sm))))
input_data_df = pd.DataFrame(input_data, columns=['sh', '0s'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)
# Compute stats for just 1 parameter
print_correlation(gal_sh_ratio,output_data)
                                                       
# Trying with all parameters 
input_data = np.column_stack((gal_sm, gal_sfr, gal_ssfr, gal_mwa, gal_hm, gal_sh_ratio))
input_data_df = pd.DataFrame(input_data, columns=['sm', 'sfr', 'ssfr', 'mwa', 'hm', 'sh_ratio'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)

# Trying with all parameters plus sfh
input_data = np.concatenate([input_data,gal_sfh],axis=1)
input_data_df = pd.DataFrame(input_data, columns=['sm', 'sfr', 'ssfr', 'mwa', 'hm', 'sh_ratio','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)




# ----- For f_core
output_data = np.copy(f_core_age)

# Trying with just 1 parameters (most correlated
input_data = np.column_stack((gal_sm,np.zeros(np.size(gal_sm))))
input_data_df = pd.DataFrame(input_data, columns=['sm', '0s'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)
# Compute stats for just 1 parameter
print_correlation(gal_sm,output_data)
                 
# Trying with all parameters 
input_data = np.column_stack((gal_sm, gal_sfr, gal_ssfr, gal_mwa, gal_hm, gal_sh_ratio))
input_data_df = pd.DataFrame(input_data, columns=['sm', 'sfr', 'ssfr', 'mwa', 'hm', 'sh_ratio'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)

# Trying with all parameters plus sfh
input_data = np.concatenate([input_data,gal_sfh],axis=1)
input_data_df = pd.DataFrame(input_data, columns=['sm', 'sfr', 'ssfr', 'mwa', 'hm', 'sh_ratio','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)




# ----- For f_vmax
output_data = np.copy(f_vmax_age)

# Trying with just 1 parameters (most correlated
input_data = np.column_stack((gal_sh_ratio,np.zeros(np.size(gal_sm))))
input_data_df = pd.DataFrame(input_data, columns=['sh_ratio', '0s'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)
# Compute stats for just 1 parameter
print_correlation(gal_sh_ratio,output_data)
                                                       
# Trying with all parameters 
input_data = np.column_stack((gal_sm, gal_sfr, gal_ssfr, gal_mwa, gal_hm, gal_sh_ratio))
input_data_df = pd.DataFrame(input_data, columns=['sm', 'sfr', 'ssfr', 'mwa', 'hm', 'sh_ratio'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)

# Trying with all parameters plus sfh
input_data = np.concatenate([input_data,gal_sfh],axis=1)
input_data_df = pd.DataFrame(input_data, columns=['sm', 'sfr', 'ssfr', 'mwa', 'hm', 'sh_ratio','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
t1,t2 = run_random_forest(input_data,input_data_df,output_data)


