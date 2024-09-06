import pickle
import matplotlib.pyplot as plt
import numpy as np

with open ('datasetFinal_linear_locus_dev-clean1-conf_sig1.pickle', 'rb') as f:
    dataset = pickle.load(f)
  
# =============================================================================
# with open ('C:/Users/Vijaya/PhD/chapter5/SignalDataset/datasetFinal_random_locus_dev-clean1-sig1.pickle', 'rb') as f:
#     dataset = pickle.load(f)
# =============================================================================

mic1_coh = np.array(dataset[0][0]['mic1_coh'])
mic2_coh = np.array(dataset[0][0]['mic2_coh'])
mic1_abs_l = np.array(dataset[0][0]['mic1_abs_l'])
mic1_abs_r = np.array(dataset[0][0]['mic1_abs_r'])
mic2_abs_l = np.array(dataset[0][0]['mic2_abs_l'])
mic2_abs_r = np.array(dataset[0][0]['mic2_abs_r'])

mic1_abs = list(np.array(mic1_abs_l) + np.array(mic1_abs_r))
mic2_abs = list(np.array(mic2_abs_l) + np.array(mic2_abs_r))


fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios':[2,2]}
        
        )

axs[0].plot(mic1_coh, c ="blue", label='D1-coh')
axs[0].plot(mic2_coh, c ="red", label='D2-coh')
axs[0].legend()
axs[0].grid()


axs[1].plot(mic1_abs, c= 'blue' , label='D1-abs')
axs[1].plot(mic2_abs,  c= 'red' , label='D2-abs')
axs[1].grid()

axs[1].legend()


# =============================================================================
# 
# def Extract(lst):
#     return [item[0] for item in lst]
# 
# locus2 = Extract(locus)
# 
# plt.plot(mic1_coh, c ="blue", label='D1 MSC')
# plt.plot(mic2_coh, c ="red", label='D2 MSC')
# 
# 
# plt.plot(mic1_abs , c ="blue", label='D1 ABS')
# plt.plot(mic2_abs, c ="red", label='D2 ABS')
# 
# 
# plt.grid()
# plt.legend()
# =============================================================================

#plt.savefig('abs-coh-randum-locus-str3.pdf')

plt.show()