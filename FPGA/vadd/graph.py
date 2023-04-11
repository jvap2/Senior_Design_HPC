import pandas as pd
import matplotlib.pyplot as plt



data=pd.read_csv('vadd_times.csv')
size=data['size'].to_numpy()
CPU=data['cpu time'].to_numpy()
GPU_Mem=data['kernel time'].to_numpy()
GPU_wo_Mem=GPU_Mem+data['total memory transfer time, total (ms)']
SU_Mem=data['speedup no mem transfer'].to_numpy()
SU_wo_Mem=data['speedup with mem transfer'].to_numpy()
through=(data['write latency (MB/s)'].to_numpy()+data['read latency (MB/s)'].to_numpy())/2
print(through)

fig, (ax1)= plt.subplots(1,1)
ax1.plot(size,CPU,'k',marker='*', label='CPU')
ax1.plot(size,GPU_Mem,'r',marker='*', label='FPGA exec')
ax1.plot(size,GPU_wo_Mem,'b',marker='*', label='FPGA total')
ax1.set_xscale("log")
ax1.grid()
ax1.legend()
ax1.set_xlabel('Vector Size')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Times for Varying Size')
plt.savefig('fig_1.png')
plt.show()

fig, (ax3)= plt.subplots(1,1)
ax3.plot(size,SU_Mem,'k',marker='*', label='FPGA total')
ax3.plot(size,SU_wo_Mem,'r',marker='*', label='FPGA exec')
ax3.set_xscale("log")
ax3.grid()
ax3.legend()
ax3.set_xlabel('Vector Size')
ax3.set_ylabel('SpeedUp')
ax3.set_title('SpeedUp for Varying Size')
plt.savefig('fig_2.png')
plt.show()

fig, (ax2)= plt.subplots(1,1)
ax2.plot(size,through,'k',marker='*', label='Throughput')
ax2.set_xscale("log")
ax2.grid()
ax3.legend()
ax2.set_xlabel('Vector Size')
ax2.set_ylabel('Throughput (MB/s)')
ax2.set_title('Throughput for Varying Size')
plt.savefig('fig_3.png')
plt.show()


