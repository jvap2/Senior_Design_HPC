import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# with open('README.md') as f:
#     rows = []
#     for row in f.readlines():
        
#         # Get rid of leading and trailing '|'
#         tmp = row[1:-2]
#         # Split line and ignore column whitespace
#         clean_line = [col.strip() for col in tmp.split('|')]
#         # Append clean row data to rows variable
#         rows.append(clean_line)
#     # Get rid of syntactical sugar to indicate header (2nd row)
#     rows = rows[:1] + rows[2:]
# print(rows)
# df = pd.DataFrame(rows)
# df.to_csv('my_file.csv', index=False, header=False)

data=pd.read_csv('my_file.csv')
Image=data['Image'].to_numpy()
CPU=data['CPU exec time(ms)'].to_numpy()
FPGA=data['FPGA exec time (ms)'].to_numpy()
GPU=data['GPU exec time(ms)']
SU_FPGA=data['SpeedUp FPGA'].to_numpy()
SU_GPU=data['SpeedUp GPU'].to_numpy()
through_FPGA=data['FPGA Throughput (MB/s)'].to_numpy()*(1/1000.0)
through_GPU=data['GPU Throughput (GB/s)'].to_numpy()

x=np.arange(len(Image))
width =.2

fig, (ax1)= plt.subplots(1,1)
ax1.bar(x-width, CPU,width, color='lightgrey', edgecolor='blue', label="CPU Exec Time")
ax1.bar(x, FPGA,width, color='k',edgecolor='white', label="FPGA Exec Time")
ax1.bar(x+width, GPU,width, color='r',edgecolor='black', label="GPU Exec Time")
# ax1.grid()
ax1.legend()
ax1.set_yscale("log")
ax1.set_xlabel('Images')
ax1.set_xticks(x)
ax1.set_xticklabels(Image)
ax1.set_ylabel('Execution Time (ms) (Log Scale)')
ax1.set_title('Execution Times for Varying Images')
plt.savefig('fig_1.png')
plt.show()

fig, (ax2)= plt.subplots(1,1)
ax2.bar(x-width/2, SU_FPGA,width,color='k',edgecolor='white', label="FPGA Speedup")
ax2.bar(x+width/2, SU_GPU,width, color='r',edgecolor='black',label="GPU Speedup")
# ax2.grid()
ax2.legend()
ax2.set_yscale("log")
ax2.set_xlabel('Images')
ax2.set_xticks(x)
ax2.set_xticklabels(Image)
ax2.set_ylabel('Speedup (Log Scale)')
ax2.set_title('Speedup for Varying Images')
plt.savefig('fig_2.png')
plt.show()

fig, (ax3)= plt.subplots(1,1)
ax3.bar(x-width/2, through_FPGA,width,color='k',edgecolor='white', label="FPGA Througput")
ax3.bar(x+width/2, through_GPU,width, color='r',edgecolor='black',label="GPU Throughput")
# ax3.grid()
ax3.legend()
ax3.set_xlabel('Images')
ax3.set_xticks(x)
ax3.set_xticklabels(Image)
ax3.set_ylabel('Throughput (GB/S)')
ax3.set_title('Throughput for Varying Images')
plt.savefig('fig_3.png')
plt.show()


