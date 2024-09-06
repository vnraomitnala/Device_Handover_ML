import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter


locusFinal = []
locusFinal2 = []
locusFinal3 = []


with open('locus_pos_list_x.pickle', 'rb') as myfile:
    locusFinal2 = pickle.load(myfile)  
 
with open('C:/Users/Vijaya/PhD/chapter5/SignalDataset/locus_pos_list_20000.pickle', 'rb') as myfile:
    locusFinal3 = pickle.load(myfile)  
    
a = []
b = []


for xx in range(len(locusFinal3)):    
    a1 = locusFinal2[xx]
    b1 = locusFinal3[xx]
    a.append(a1)
    b.append(b1)
    
#first_and_second_column_vals1 = list(map(itemgetter(0, 1), b[199]))
first_and_second_column_vals2 = list(map(itemgetter(0, 1), b[150]))

standardRoom_source_locs2 = [ [ 0.5 +i*0.12] for i in range(50) ]   
standardRoom_source_locs21 = [ 2 for i in range(50) ]

   
plt.scatter(1.0, 3.5, marker = 'X', color = 'k', s= 200)
plt.scatter(5.5, 3.5, marker = 'X', color = 'k', s= 200)
plt.text(1.2, 3.5, 'D1', fontsize=13, fontweight='bold')
plt.text(5.7, 3.5, 'D2', fontsize=13, fontweight='bold')

plt.plot(standardRoom_source_locs2, standardRoom_source_locs21 , 'k',  linestyle = 'dotted', linewidth = 4.0)
# =============================================================================
# plt.text(1.5, 2.2, 'l', fontsize=15, fontweight='bold')
# =============================================================================

#plt.plot( standardRoom_source_locs2, first_and_second_column_vals1, 'r',  linewidth = 3.0, linestyle = 'dotted')
# =============================================================================
# plt.text(0.5, 4.5, 'r1', fontsize=15, fontweight='bold')
# plt.text(3.7, 2.5, 'r2', fontsize=15, fontweight='bold')
# plt.text(3.7, 3.7, 'r4', fontsize=15, fontweight='bold')
# plt.text(1.2, 1.1, 'r4', fontsize=15, fontweight='bold')
# =============================================================================

plt.plot(standardRoom_source_locs2, first_and_second_column_vals2, 'g',  linewidth = 3.0, linestyle = 'dotted')


plt.plot(7, 5.5)

# =============================================================================
# 
# plt.plot(a[1], b[1], 'r',  linewidth = 3.0, linestyle = 'dotted')
# plt.plot(a[2], b[2], 'g',  linewidth = 3.0, linestyle = 'dotted')
# plt.plot(a[3], b[300], 'm',  linewidth = 3.0, linestyle = 'dotted')
# =============================================================================

# =============================================================================
#plt.plot(a[130][50:], b[150], 'c',  linewidth = 3.0, linestyle = 'dotted')
# plt.plot(a[110], b[199]["locus"], 'r',  linewidth = 3.0, linestyle = 'dotted')
# =============================================================================



plt.grid()
# =============================================================================
# legend = plt.legend(['D1', 'D2', 'L', 'R1'], loc= 'lower right')
# legend.FontSize = 2
# =============================================================================

plt.savefig('linear_locus_pos-latest2.pdf')
plt.show()    
    

