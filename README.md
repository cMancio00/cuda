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
![alt text](cuda_indexing.png "How the GPU can split the work on an array")


### How to call funtions that will run on the GPU
We have a special syntax to run functions on GPU which is `myFunction<<< #blocks , #threads >>>(parameters_of_the_function)`
For example the code below will run the function myFunction using 16 blocks of 128 threads each with no parameters.
```c++
myFunction<<< 16, 128 >>>()
```