### Matrix Multiplication Performance Estimates- Hardware Emulation
|Rows|Columns| CPU times (ms) | Naive Kernel time (ms) |Partition Kernel time (ms) | Memory Transfer Time, Write (ms) | Memory Transfer Time, Read (ms) | Total Memory Transfer Time (ms) | Naive Speedup| Partitition Speedup|
|----|----|---------------|---------------|---------------|--------------|--------------|--------------|----------|----------|
|64|64|0.96408|0.487419|0.101968|4.352|31.556|35.908|1.9779|9.4547|
|256|256|49.199|
|512|512|640.299|
|1024|1024|6204.88|
|2048|2048|151171.0|


### Matrix Multiplication Hardware Runs
|Rows|Columns| CPU times (ms) | Naive Kernel time (ms) |Partition Kernel time (ms) | Memory Transfer Time, Write (ms) | Memory Transfer Time, Read (ms) | Total Memory Transfer Time (ms) | Naive Speedup| Partitition Speedup|
|----|----|---------------|---------------|---------------|--------------|--------------|--------------|----------|----------|
|128|128|0.96408|3.746|0.402|
|256|256|49.199|36.28|1.989|
|512|512|640.299|
|1024|1024|6204.88|
|2048|2048|151171.0|
