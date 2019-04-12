# Open data for vespa outputs run on LGalaxies data
def load_vespa_lgal_data():
    
    # Import modules
    import numpy as np, pandas as pd
    
    # ----- Load data for individual parameters
    observables_data = pd.read_csv('/Users/dominicbates/Documents/Python/Halo Formation/data/vespa_lgal_data/results_BCGnoisemodel_withdust__out.txt')
    # Extract individual arrays from database
    sm_v = np.array(observables_data['M_vespa']) # Stellar mass
    sm_l = np.array(observables_data['M_Lgal'])
    mwa_v = np.array(observables_data['MWA_vespa'])  # Mass-weighted age
    mwa_l = np.array(observables_data['MWA_Lgal'])
    ysf_v = np.array(observables_data['youngFrac_vespa'])  # young stellar fraction
    ysf_l = np.array(observables_data['youngFrac_Lgal'])
    # Delete database from memory
    del observables_data
    
    # ----- Load sfh bins data
    sfh_bins_v = np.array([0.011, 0.025, 0.039, 0.061, 0.0945, 0.146, 0.226, 0.35, 0.5415, 0.839, 1.295, 2.005, 3.11, 4.81, 7.44, 11.52])
    sfh_bins_l = np.array([0.01027,0.0308099, 0.0616199, 0.123240, 0.205399, 0.328639, 0.576556, 1.08082, 1.76056, 2.78558, 4.75909, 7.87901, 10.7796])
    
    # ----- Load sfh data (have to do this because file is space seperated 
    # AND different row lengths - i.e. 1st line is vespa, 2nd is lgal...)
    sfh_data_v = np.zeros([999,16])
    sfh_data_l = np.zeros([999,13])
    # Ignore first 2 lines
    count = -2
    # Open file and loop through to open data
    with open('/Users/dominicbates/Documents/Python/Halo Formation/data/vespa_lgal_data/SFH_data.txt') as tsv:        
        for line in tsv:
            if count>=0:
                # If odd line number (e.g. 1, 3, 5...)
                if ((count % 2) == 0):
                    sfh_data_v[int(count/2),:] = np.array(line.split(),dtype='float')
                else:
                    sfh_data_l[int((count/2)-0.5),:] = np.array(line.split(),dtype='float')
            count+=1
            
    # Interpolate LGALAXIES data to same shape as VESPA data
    sfh_data_l_vbins = np.zeros(np.shape(sfh_data_v))
    for n in range(np.size(sfh_data_v[:,0])):
        # Interpolate to vespa data shape
        sfh_data_l_vbins[n] = np.interp(sfh_bins_v, sfh_bins_l, sfh_data_l[n])
        # Normalise to 1 again
        sfh_data_l_vbins[n] = sfh_data_l_vbins[n]/(sfh_data_l_vbins[n].sum())
        
        
    # Interpolate VESPA data to same shape as LGALAXIES data
    sfh_data_v_lbins = np.zeros(np.shape(sfh_data_l))
    for n in range(np.size(sfh_data_l[:,0])):
        # Interpolate to vespa data shape
        sfh_data_v_lbins[n] = np.interp(sfh_bins_l, sfh_bins_v, sfh_data_v[n])
        # Normalise to 1 again
        sfh_data_v_lbins[n] = sfh_data_v_lbins[n]/(sfh_data_v_lbins[n].sum())
    
    
    return sm_l, mwa_l, ysf_l, sfh_data_l, sfh_data_l_vbins, sfh_bins_l, \
           sm_v, mwa_v, ysf_v, sfh_data_v, sfh_data_v_lbins, sfh_bins_v


# Compute covariance in errors beween different observables measured by vespa
def compute_covariance_vespa_lgal_data():

    # Import modules
    import numpy as np
    
    # Load data for LGalaxies + Vespa run on LGalaxies data
    sm_l, mwa_l, ysf_l, sfh_l, sfh_l_vbins, sfh_bins_l, \
    sm_v, mwa_v, ysf_v, sfh_v, sfh_v_lbins, sfh_bins_v = \
               load_vespa_lgal_data()
               
    # Compute offset for all parameters (i.e. vespa - true)
    off_sm = sm_v - sm_l
    off_mwa = mwa_v - mwa_l
    off_ysf = ysf_v - ysf_l
    off_sfh_vbins = sfh_v - sfh_l_vbins
    
    # Combine data to single file (row = each galaxy, column = sm,mwa,ysf,sfh1,sfh2...)
#    comb_params = np.swapaxes(np.array([sm_l,mwa_l,ysf_l]),0,1)
#    comb_all = np.concatenate((comb_params,sfh_l),axis=1)
    
    # Combine offsets to single file (same definition)
    off_params = np.swapaxes(np.array([off_sm,off_mwa,off_ysf]),0,1)
    off_all = np.concatenate((off_params,off_sfh_vbins),axis=1)
    
    # Compute covariance between these offsets (need to flip axis)
    cov_all = np.cov(off_all,rowvar=False)
    
    # Compute mean offset 
    off_means = np.mean(off_all,axis=0)
    
    # Test compute errors 
    #test_errors = np.random.multivariate_normal(off_means,cov_all,size=100000)
    
    # Return covariance between offsets, and mean value of offset for each param
    return cov_all, off_means
    # To compute random errors with this covariance:
    #   np.random.multivariate_normal(off_means,cov_all,size=1000)




#test_errors = np.random.multivariate_normal(off_means,cov_all,size=100000)
#       
#names = ['$eM_{\star}$','$eMWA$','$eYSF$', 
#         '$e1$','$e2$','$e3$','$e4$','$e5$','$e6$','$e7$','$e8$',
#         '$e9$','$e10$','$e11$','$e12$','$e13$','$e14$','$e15$','$e16$']
#
#import pygtc
#pygtc.plotGTC(test_errors,paramNames=names,plotName='test_covariance_plot.pdf')

