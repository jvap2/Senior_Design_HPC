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
size=data['Size'].to_numpy()
CPU=data['CPU Exec time (ms)'].to_numpy()
GPU_wo_Mem=data[' GPU Exec time (ms) '].to_numpy()
DH=data[' Memory Transfer (D->H)'].to_numpy()
HD=data[' Memory Transfer (H->D) '].to_numpy()
GPU_Mem=GPU_wo_Mem+DH+HD
SU_Mem=np.divide(CPU,GPU_Mem)
SU_wo_Mem=np.divide(CPU,GPU_wo_Mem)
through=data[' Throughput (GB/s)'].to_numpy()
print(through)

fig, (ax1)= plt.subplots(1,1)
ax1.plot(size,CPU, label='CPU')
ax1.plot(size,GPU_Mem, label='GPU total')
ax1.plot(size,GPU_wo_Mem, label='GPU exec')
ax1.set_xscale("log")
ax1.grid()
ax1.legend()
ax1.set_xlabel('Vector Size')
ax1.set_ylabel('Execution Time')
ax1.set_title('Execution Times for Varying Size')
plt.savefig('fig_1.png')
plt.show()

fig, (ax3)= plt.subplots(1,1)
ax3.plot(size,SU_Mem, label='GPU total')
ax3.plot(size,SU_wo_Mem, label='GPU exec')
ax3.set_xscale("log")
ax3.grid()
ax3.legend()
ax3.set_xlabel('Vector Size')
ax3.set_ylabel('SpeedUp')
ax3.set_title('SpeedUp for Varying Size')
plt.savefig('fig_2.png')
plt.show()

fig, (ax2)= plt.subplots(1,1)
ax2.plot(size,through)
ax2.set_xscale("log")
ax2.grid()
ax2.set_xlabel('Vector Size')
ax2.set_ylabel('Throughput (GB/s)')
ax2.set_title('Throughput for Varying Size')
plt.savefig('fig_3.png')
plt.show()


