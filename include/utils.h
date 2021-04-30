#ifndef UTILS_H
#define UTILS_H

#include <cstdint> 	/* int64_t, uint64_t */
#include <cstdlib>	/* RAND_MAX */

/**
 * Return vector with each element of the input at its bit-reversed position
 *
 * @param vec The vector to bit reverse
 * @param n   The length of the vector, must be a power of two
 * @return    The bit reversed vector
 */
uint64_t *bit_reverse(uint64_t *vec, uint64_t n);

/**
 * Compare two vectors element-wise and return whether they are equivalent
 *
 * @param vec1	The first vector to compare
 * @param vec2 	The second vector to compare
 * @param n 	The length of the vectors
 * @param debug	Whether to print debug information (will run entire vector)
 * @return 	Whether the two vectors are element-wise equivalent
 */
bool compVec(uint64_t *vec1, uint64_t *vec2, uint64_t n, bool debug=false);

/**
 * Perform the operation 'base^exp (mod m)' using the memory-efficient method
 *
 * @param base	The base of the expression
 * @param exp	The exponent of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m);

/**
 * Perform the operation 'base (mod m)'
 *
 * @param base	The base of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
uint64_t modulo(int64_t base, int64_t m);

/**
 * Print an array of arbitrary length in a readable format
 *
 * @param vec	The array to be displayed
 * @param n	The length of the array
 */
void printVec(uint64_t *vec, uint64_t n);

/**
 * Generate an array of arbitrary length containing random positive integers 
 *
 * @param n	The length of the array
 * @param max	The maximum value for an array element [Default: RAND_MAX]
 */
uint64_t *randVec(uint64_t n, uint64_t max=RAND_MAX);

#endif
