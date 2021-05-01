#include <cmath>		/* log2(), pow() */
#include <cstdint>		/* uint64_t */
#include <cstdlib> 		/* malloc() */
#include <iostream>

#include "../include/utils.cuh"	/* bit_reverse(), modExp(), modulo() */
#include "../include/ntt.h" 	//INCLUDE HEADER FILE


__global__ void blockComputation(uint64_t*,uint64_t*,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t**) ;
void blockComp(uint64_t* , uint64_t ,uint64_t,uint64_t ,uint64_t,uint64_t**) ;

void cpuToGpuMemcpy(uint64_t* h_data,uint64_t* d_data,int size)
{
    cudaError_t err = cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice) ;
    if(err != cudaSuccess)
    {
            fprintf(stderr,"Failed to copy vector from host device!",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
    }
}

void gpuToCpuMemcpy(uint64_t* d_data,uint64_t* h_data,int size)
{
    cudaError_t err = cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost) ;
    if(err != cudaSuccess)
    {
            fprintf(stderr,"Failed to copy vector from gpu device!",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
    }
    cudaFree(d_data) ;
}


/**
 * Perform an in-place iterative breadth-first decimation-in-time Cooley-Tukey NTT on an input vector and return the result
 *
 * @param vec 	The input vector to be transformed
 * @param n	The size of the input vector
 * @param p	The prime to be used as the modulus of the transformation
 * @param r	The primitive root of the prime
 * @param rev	Whether to perform bit reversal on the input vector
 * @return 	The transformed vector
 */



uint64_t *inPlaceNTT_DIT(uint64_t *vec, uint64_t n, uint64_t p, uint64_t r, uint64_t** twiddleFactors, bool rev){

	uint64_t *result,*result_cpu;

	uint64_t m, k_, a ;
        uint64_t factor1, factor2 ;
	result = (uint64_t *) malloc(n*sizeof(uint64_t));
	result_cpu = (uint64_t *) malloc(n*sizeof(uint64_t));

	if(rev){
		result = bit_reverse(vec, n);
		result_cpu = bit_reverse(vec, n);
	}else{
		for(uint64_t i = 0; i < n; i++){	
			result[i] = vec[i];
			result_cpu[i] = vec[i];
		}
	}


	for(uint64_t i = 1; i <= log2(n); i++){ 

		m = pow(2,i);
		k_ = (p - 1)/m;
		a = modExp(r,k_,p);
		//std:;cout<<"\na = "<<a ;

        
		for(uint64_t j = 0; j < n; j+=m){

			for(uint64_t k = 0; k < m/2; k++){

				factor1 = result_cpu[j + k];
				factor2 = modulo(modExp(a,k,p)*result_cpu[j + k + m/2],p);
			
				result_cpu[j + k] 	= modulo(factor1 + factor2, p);
				result_cpu[j + k+m/2] 	= modulo(factor1 - factor2, p);
				//if(j==0)
				//	std::cout<<"\nmod(a,k,p) = "<<a<<","<<k<<","<<modExp(a,k,p) ;
			}
		}
        blockComp(result,n,m,i,p,twiddleFactors) ;

	}
	bool compCPUGPUResult = compVec(result,result_cpu,n,true) ;
	std::cout<<"\nComparing output of cpu and gpu :"<<compCPUGPUResult ;
	return result;

}

void blockComp(uint64_t* res, uint64_t resLength,uint64_t blockSize,uint64_t a,uint64_t p,uint64_t** twiddleFactors)
{
    uint64_t *cuda_result, *cuda_output  ;
    uint64_t sizeOfRes = resLength*sizeof(uint64_t) ;
    cudaMalloc(&cuda_result,sizeOfRes) ;
	cudaMalloc(&cuda_output,sizeOfRes) ;
    cpuToGpuMemcpy(res,cuda_result,sizeOfRes) ;

    int tpb = 32;//blockSize;
    int bpg = (resLength -1 + tpb)/tpb ;

    
    blockComputation<<<bpg,tpb>>>(cuda_result,cuda_output,resLength,blockSize,a,p,twiddleFactors) ;
    cudaError_t err = cudaGetLastError() ;

	if(err != cudaSuccess)
	{
	    fprintf(stderr,"Issues in running the kernel",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
	}

    gpuToCpuMemcpy(cuda_output,res,sizeOfRes) ;
	cudaFree(cuda_result) ;
}

__global__ void blockComputation(uint64_t* result, uint64_t* output,uint64_t n,uint64_t m,uint64_t a,uint64_t p,uint64_t** twiddleFactors)
{
    
    
    //uint64_t k=threadIdx.x ;
    uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x ;
    uint64_t k ;
    uint64_t factor1,factor2 ;
    //if(idx < (n-(n%m)))
    if(idx < n)
	{
        //j = idx/m ;
	k = idx%m ;
	if(k < m/2)
	{
		factor1 = result[idx] ;
		//factor2 = modulo(modExp(a,k,p)*result[idx+m/2],p);
		factor2 = modulo(twiddleFactors[a][k]*result[idx+m/2],p);	
		output[idx] = modulo(factor1+factor2,p) ;
	}
	else
	{
		factor1 = result[idx - m/2] ;
		//factor2 = modulo(modExp(a,k-(m/2),p)*result[idx],p) ;
		factor2 = modulo(twiddleFactors[a][k-(m/2)]*result[idx],p) ;
		output[idx] = modulo(factor1-factor2,p) ;
	}
    }
    /*
    uint64_t factor1,factor2 ;
	if(j < n){
        if(k < m/2){
            factor1 = result[j + k];
		    factor2 = modulo(modExp(a,k,p)*result[j + k + m/2],p);	
		    result[j + k] 		= modulo(factor1 + factor2, p);
		    result[j + k+m/2] 	= modulo(factor1 - factor2, p);
        }
    }
    */
}
