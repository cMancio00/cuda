#include <stdio.h>

__global__
void loop()
{
    printf("This is iteration number %d\n", (blockDim.x * blockIdx.x)+threadIdx.x);
}

int main()
{
  loop<<<2,5>>>();
  cudaDeviceSynchronize();
}
