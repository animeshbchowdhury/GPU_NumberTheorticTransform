#include <cmath>		/* pow() */
#include <cstdint>		/* uint64_t */
#include <ctime>		/* time() */

#include <unistd.h>
#include <iostream>

#include "../include/ntt.h"	/* naiveNTT(), outOfPlaceNTT_DIT() */
#include "../include/utils.h"	/* printVec() */

using namespace std;

int main(int argc, char *argv[]){
  uint64_t n = 4096;
  uint64_t p = 68719403009;
  uint64_t r = 36048964756;

	uint64_t vec[n];

  for (int i = 0; i < n; i++){
    vec[i] = i;
  }

  uint64_t *outVec = inPlaceNTT_DIT(vec,n,p,r);

	printVec(outVec, n);

	return 0;

}
