#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
__global__ void vectorAdd(float *,float*,float*,int) ;
__global__ 
void vectorAdd(float* A, float* B, float* C, int n)
{
	int i=threadIdx.x+blockDim.x*blockIdx.x ;
	if(i<n)
		C[i] = A[i]+B[i] ;
}

void gpuMalloc(float *d_data,int size)
{
    cudaError_t err = cudaMalloc(&d_data,size) ;
    if(err != cudaSuccess)
    {
	   fprintf(stderr,"Failed to allocate memory (error code %s)!\n",cudaGetErrorString(err)) ;
	   exit(EXIT_FAILURE) ;
    } 
}

void cpuToGpuMemcpy(float* h_data,float* d_data,int size)
{
    cudaError_t err = cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice) ;
    if(err != cudaSuccess)
    {
	    fprintf(stderr,"Failed to copy vector from host device!",cudaGetErrorString(err)) ;
    	    exit(EXIT_FAILURE) ;
    }
}

void gpuToCpuMemcpy(float* d_data,float* h_data,int size)
{
    cudaError_t err = cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost) ;
    if(err != cudaSuccess)
    {
            fprintf(stderr,"Failed to copy vector from gpu device!",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
    }
    cudaFree(d_data) ;
}

void vecAdd(float* h_A,float *h_B, float* h_C, int n)
{
	int size = n * sizeof(float) ;
	float* d_A = NULL, *d_B = NULL, *d_C = NULL ;
	//gpuMalloc(d_A,size) ;
	cudaMalloc(&d_A,size) ;
	cudaMalloc(&d_B,size) ;
	cudaMalloc(&d_C,size) ;
	cpuToGpuMemcpy(h_A,d_A,size) ;
	cpuToGpuMemcpy(h_B,d_B,size) ;
	cpuToGpuMemcpy(h_C,d_C,size) ;


	int tpb = 128 ;
	int bpg = (n+tpb-1)/tpb ;
	vectorAdd<<<bpg,tpb>>>(d_A,d_B,d_C,n) ;
	cudaError_t err = cudaGetLastError() ;

	if(err != cudaSuccess)
	{
	    fprintf(stderr,"Issues in running the kernel",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
	}
	gpuToCpuMemcpy(d_A,h_A,size) ;
	gpuToCpuMemcpy(d_B,h_B,size) ;
	gpuToCpuMemcpy(d_C,h_C,size) ;
}


int main()
{
	int n = 10;
	float *h_A = (float *)calloc(n,sizeof(float)) ;
	float *h_B = (float *)calloc(n,sizeof(float)) ;
	float *h_C = (float *)calloc(n,sizeof(float)) ;
	for(int i=0;i<10;i++)
	{
		h_A[i] = (float)i ;
		h_B[i] = (float)i ;
	}
	vecAdd(h_A,h_B,h_C,n) ;
	printf("\nResult:\n") ;
	for(int i=0;i<10;i++)
	   printf(",%f",h_C[i]) ;

	return 0 ;
}
