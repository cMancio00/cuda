# CUDA

## Fundametal concepts

### How to run functions on GPU

The keyword to declare kernel functions (the ones that can be run on gpu) is `__global__`.
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
