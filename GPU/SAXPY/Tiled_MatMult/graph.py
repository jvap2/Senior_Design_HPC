import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open('README.md') as f:
    rows = []
    for row in f.readlines():
        
        # Get rid of leading and trailing '|'
        tmp = row[1:-2]
        # Split line and ignore column whitespace
        clean_line = [col.strip() for col in tmp.split('|')]
        # Append clean row data to rows variable
        rows.append(clean_line)
    # Get rid of syntactical sugar to indicate header (2nd row)
    rows = rows[:1] + rows[2:]
print(rows)
df = pd.DataFrame(rows)
df.to_csv('my_file.csv', index=False, header=False)

data=pd.read_csv('my_file.csv')
size=data['Rows'].to_numpy()
CPU=data['CPU time (ms)'].to_numpy()
GPU_wo_Mem=data['Tiled GPU time (ms)'].to_numpy()
GPU_Mem=GPU_wo_Mem+data['Memory Transfer (ms)'].to_numpy()
GPU_wo_Mem_N=data['Non-Tiled GPU time (ms)'].to_numpy()
GPU_Mem_N=GPU_wo_Mem+data['Memory Transfer (ms)'].to_numpy()
through=np.divide(3*np.multiply(size,size),data['Memory Transfer (ms)'].to_numpy())*10e-6
SU_Mem=np.divide(CPU,GPU_Mem)
SU_wo_Mem=np.divide(CPU,GPU_wo_Mem)
SU_Mem_N=np.divide(CPU,GPU_Mem_N)
SU_wo_Mem_N=np.divide(CPU,GPU_wo_Mem_N)
print(through)
print(SU_Mem)

fig, (ax1)= plt.subplots(1,1)
ax1.plot(size,CPU, 'k',label='CPU')
ax1.plot(size,GPU_Mem_N, 'b',label='Non-Tiled GPU total')
ax1.plot(size,GPU_wo_Mem_N,'r', label='Non-TiledGPU exec')
ax1.set_xscale("log")
ax1.grid()
ax1.legend()
ax1.set_xlabel('Matrix Dimension')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Times for Varying Size, Non-Tiled')
plt.savefig('fig_1.png')
plt.show()

fig, (ax5)= plt.subplots(1,1)
ax5.plot(size,CPU, 'k',label='CPU')
ax5.plot(size,GPU_Mem, 'k',marker='*',label='Tiled GPU total')
ax5.plot(size,GPU_wo_Mem,'c', marker='*',label='Tiled GPU exec')
ax5.set_xscale("log")
ax5.grid()
ax5.legend()
ax5.set_xlabel('Matrix Dimension')
ax5.set_ylabel('Execution Time (ms)')
ax5.set_title('Execution Times for Varying Size, Tiled')
plt.savefig('fig_2.png')
plt.show()

fig, (ax3)= plt.subplots(1,1)
ax3.plot(size,SU_Mem_N,'b', label='Non-Tiled GPU total')
ax3.plot(size,SU_wo_Mem_N,'r', label='Non-Tiled GPU exec')
ax3.set_xscale("log")
ax3.grid()
ax3.legend()
ax1.set_xlabel('Matrix Dimension')
ax3.set_ylabel('SpeedUp')
ax3.set_title('SpeedUp for Varying Size, Non-Tiled')
plt.savefig('fig_3.png')
plt.show()

fig, (ax4)= plt.subplots(1,1)
ax4.plot(size,SU_Mem,'k', marker='*', label='Tiled GPU total')
ax4.plot(size,SU_wo_Mem,'c', marker='*',label='Tiled GPU exec')
ax4.set_xscale("log")
ax4.grid()
ax4.legend()
ax1.set_xlabel('Matrix Dimension')
ax4.set_ylabel('SpeedUp')
ax4.set_title('SpeedUp for Varying Size,Tiled')
plt.savefig('fig_4.png')
plt.show()

fig, (ax2)= plt.subplots(1,1)
ax2.plot(size,through)
ax2.set_xscale("log")
ax2.grid()
ax1.set_xlabel('Matrix Dimension')
ax2.set_ylabel('Throughput (GB/s)')
ax2.set_title('Throughput for Varying Size')
plt.savefig('fig_5.png')
plt.show()


