#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <cmath>		/* log2() */
#include <cstdint> 		/* int64_t, uint64_t */
#include <cstdlib>		/* srand(), rand() */
#include <ctime>		/* time() */
#include <iostream> 		/* std::cout, std::endl */

#include "../include/utils.cuh" 	//INCLUDE HEADER FILE
/*
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
*/

uint64_t** preComputeTwiddleFactor(uint64_t n,uint64_t p, uint64_t r)
{
	uint64_t x,y ;
	uint64_t m,a,k_ ;
	uint64_t** twiddleFactorMatrix = (uint64_t**)malloc((log2(n))*sizeof(uint64_t*)) ;
	for(x=0;x < log2(n);x++){
		twiddleFactorMatrix[x] = (uint64_t*)calloc((n/2)*sizeof(uint64_t)) ;
		m = pow(2,x+1) ;
		k_ = (p-1) / m ;
		a = modExp(r,k_,p) ;
		for(y=0;y<m/2;y++)
			twiddleFactorMatrix[x][y] = modExp(a,y,p) ;
	}
	return twiddleFactorMatrix ;
}


bool compVec(uint64_t *vec1, uint64_t *vec2, uint64_t n, bool debug){

	bool comp = true;
	for(uint64_t i = 0; i < n; i++){

		if(vec1[i] != vec2[i]){
			comp = false;

			if(debug){
				std::cout << "(vec1[" << i << "] : " << vec1[i] << ")";
				std::cout << "!= (vec2[" << i << "] : " << vec2[i] << ")";
				std::cout << std::endl;
			}else{
				break;
			}
		}
	}

	return comp;
}

/**
 * Return vector with each element of the input at its bit-reversed position
 *
 * @param vec The vector to bit reverse
 * @param n   The length of the vector, must be a power of two
 * @return    The bit reversed vector
 */
uint64_t *bit_reverse(uint64_t *vec, uint64_t n){

	uint64_t num_bits = log2(n);

	uint64_t *result;
	result = (uint64_t *) malloc(n*sizeof(uint64_t));

	uint64_t reverse_num;
	for(uint64_t i = 0; i < n; i++){

		reverse_num = 0;
		for(uint64_t j = 0; j < num_bits; j++){

			reverse_num = reverse_num << 1;
			if(i & (1 << j)){
				reverse_num = reverse_num | 1;
			}
		}

		result[reverse_num] = vec[i];

	}

	return result;
}


/**
 * Perform the operation 'base^exp (mod m)' using the memory-efficient method
 *
 * @param base	The base of the expression
 * @param exp	The exponent of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__host__ __device__ uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m){

	uint64_t result = 1;
	
	while(exp > 0){

		if(exp % 2){

			result = modulo(result*base, m);

		}

		exp = exp >> 1;
		base = modulo(base*base,m);
	}

	return result;
}

/**
 * Perform the operation 'base (mod m)'
 *
 * @param base	The base of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__host__ __device__ uint64_t modulo(int64_t base, int64_t m){
	int64_t result = base % m;

	return (result >= 0) ? result : result + m;
}

/**
 * Print an array of arbitrary length in a readable format
 *
 * @param vec	The array to be displayed
 * @param n	The length of the array
 */
void printVec(uint64_t *vec, uint64_t n){

	std::cout << "[";
	for(uint64_t i = 0; i < n; i++){

		std::cout << vec[i] << ",";

	}
	std::cout << "]" << std::endl;
}

/**
 * Generate an array of arbitrary length containing random positive integers 
 *
 * @param n	The length of the array
 * @param max	The maximum value for an array element [Default: RAND_MAX]
 */
uint64_t *randVec(uint64_t n, uint64_t max){

	uint64_t *vec;
	vec = (uint64_t *)malloc(n*sizeof(uint64_t));

	srand(time(0));
	for(uint64_t i = 0; i < n; i++){

		vec[i] = rand()%(max + 1);

	}

	return vec;
}

