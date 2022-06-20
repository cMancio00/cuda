# CUDA

## Fundametal concepts

### How to declare functions that will run on GPU

The keyword to declare kernel functions (the ones that can be run on gpu) is `__global__`.
This will tell the CUDA C++ compiler that the function will run on the GPU and it can be called from CPU code.
For example the code below shows how to use the keyword
``` c++
__global__
void myProcedure()
{
    .
    .
    . some code here
    .
}
```

### How to allocate memory on the GPU
We can allocate memory on the GPU (device) but it can be also accessible from the CPU (host).
The command is similar to the c function `malloc`.
The syntax is `cudaMallocManaged(&pointer, size_to_allocate)`
We have to pass a pointer's address and the size that we want to allocate. The function will return the inizialized pointer.
```c++
int *x;
cudaMallocManaged(&x, sizeof(int));
// x will point to a memory location of the first allocated byte
```

### How to deallocate memory on the GPU
We can also deallocate memory by using `cudaFree(what_to_deallocate)`
```c++
int *x;
cudaMallocManaged(&x, sizeof(int));
// now we have memory allocated

cudaFree(x);
// now we don't have memory allocated
```
### How the GPU splits the work
The GPU can achive massive parallelism performance if we take advantage of the number of cores.
We split the work in the following way. Grid -> Block -> Thread
The Grid is the major block. We can have many grids, depending of what GPU we are using. The important thing is that the number of grids must be a power of 2.
In every grid we have blocks that must also be a power of 2.
And finally in every block we have threads, also a power of 2.
The picture below shows a division of the work on an array and how to retrive the index of the elements.
![alt text](cuda_indexing.png "How the GPU can split the work on an array")

### How to call funtions that will run on the GPU
We have a special syntax to run functions on GPU which is `myFunction<<< #blocks , #threads >>>(parameters_of_the_function)`
For example the code below will run the function myFunction using 16 blocks of 128 threads each with no parameters.
```c++
myFunction<<< 16, 128 >>>()
```
If we want to know how many blocks we need given the number of threads (or the dimension of the block) we can use
``` c++
int blockSize = 256;
int numBlocks = (len_of_array + blockSize - 1) / blockSize;
```

### How to synchronize the host and the device
When lunching the CUDA kernel will not block the calling CPU thread, so we have to say the CPU to wait for the GPU to finish the computation before accessing the result. The funcion to do that is ` cudaDeviceSynchronize()`