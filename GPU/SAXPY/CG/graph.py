import pandas as pd
import matplotlib.pyplot as plt

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
size=data['SIZE'].to_numpy()
CPU=data['CPU Execution Time (ms)'].to_numpy()
GPU_Mem=data['GPU Execution Time W/ Mem Transfer(ms)'].to_numpy()
GPU_wo_Mem=data['GPU Execution Time W/O Mem Transfer(ms)'].to_numpy()
SU_Mem=data['SpeedUp (W/ Mem)'].to_numpy()
SU_wo_Mem=data['SpeedUp (W/O Mem)'].to_numpy()
through=data['Throughput (GB/s)'].to_numpy()
print(through)

fig, (ax1)= plt.subplots(1,1)
ax1.plot(size[:6],CPU[:6],'k',marker='*', label='CPU')
ax1.plot(size[:6],GPU_Mem[:6],'r',marker='*', label='GPU total')
ax1.plot(size[:6],GPU_wo_Mem[:6],'b',marker='*', label='GPU exec')
ax1.plot(size[5:],CPU[5:],'k',marker='d', label='CPU, Linear Operator')
ax1.plot(size[5:],GPU_Mem[5:],'r',marker='d', label='GPU total, Linear Operator')
ax1.plot(size[5:],GPU_wo_Mem[5:],'b',marker='d', label='GPU exec, Linear Operator')
ax1.set_xscale("log")
ax1.grid()
ax1.legend()
ax1.set_xlabel('Vector Size')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Times for Varying Size')
plt.savefig('fig_1.png')
plt.show()

fig, (ax3)= plt.subplots(1,1)
ax3.plot(size[:6],SU_Mem[:6],'k',marker='*', label='GPU total')
ax3.plot(size[:6],SU_wo_Mem[:6],'r',marker='*', label='GPU exec')
ax3.plot(size[5:],SU_Mem[5:],'k', marker='d',label='GPU total, Linear Operator')
ax3.plot(size[5:],SU_wo_Mem[5:],'r',marker='d', label='GPU exec, Linear Operator')
ax3.set_xscale("log")
ax3.grid()
ax3.legend()
ax3.set_xlabel('Vector Size')
ax3.set_ylabel('SpeedUp')
ax3.set_title('SpeedUp for Varying Size')
plt.savefig('fig_2.png')
plt.show()

fig, (ax2)= plt.subplots(1,1)
ax2.plot(size[:6],through[:6],'k',marker='*', label='Throughput')
ax2.plot(size[5:],through[5:],'r',marker='d', label='Throughput, Linear Operator')
ax2.set_xscale("log")
ax2.grid()
ax3.legend()
ax2.set_xlabel('Vector Size')
ax2.set_ylabel('Throughput (GB/s)')
ax2.set_title('Throughput for Varying Size')
plt.savefig('fig_3.png')
plt.show()


