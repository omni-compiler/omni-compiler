/*
 *   COARRAY GET SUBROUTINE VERSION (for optimization)
 *
 */

#include "xmpf_internal_coarray.h"


/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_getsub_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char **localAddr, int *rank,
                                    int skip[], int skip_local[], int count[])
{
  XMPCO_GET_arrayStmt(*descPtr, *baseAddr, *element,
                      *coindex, *localAddr, *rank,
                      skip, skip_local, count);
}


/***************************************************\
    entry for error messages
\***************************************************/

void xmpf_coarray_getsub_err_len_(void **descPtr,
                                  int *len_mold, int *len_src)
{
  char *name = _XMPCO_get_nameOfCoarray(*descPtr);

  _XMPCO_debugPrint("ERROR DETECTED: xmpf_coarray_getsub_err_len_\n"
                          "  coarray name=\'%s\', len(mold)=%d, len(src)=%d\n",
                          name, *len_mold, *len_src);

  _XMPCO_fatal("mismatch length-parameters found in "
                     "optimized get-communication on coarray \'%s\'", name);
}


void xmpf_coarray_getsub_err_size_(void **descPtr, int *dim,
                                   int *size_mold, int *size_src)
{
  char *name = _XMPCO_get_nameOfCoarray(*descPtr);

  _XMPCO_debugPrint("ERROR DETECTED: xmpf_coarray_getsub_err_size_\n"
                          "  coarray name=\'%s\', i=%d, size(mold,i)=%d, size(src,i)=%d\n",
                          name, *dim, *size_mold, *size_src);

  _XMPCO_fatal("Mismatch sizes of %d-th dimension found in "
                     "optimized get-communication on coarray \'%s\'", *dim, name);
}


