/*
 *   COARRAY GET
 *
 */

#include "xmpf_internal_coarray.h"


/***************************************************\
    entry
\***************************************************/

void xmpf_coarray_get_scalar_(void **descPtr, char **baseAddr, int *element,
                              int *coindex, char *result)
{
  XMPCO_GET_scalarExpr(*descPtr, *baseAddr, *element, *coindex, result);
}



void xmpf_coarray_get_array_(void **descPtr, char **baseAddr, int *element,
                             int *coindex, char *result, int *rank,
                             int skip[], int count[])
{
  return XMPCO_GET_arrayExpr(*descPtr, *baseAddr, *element,
                             *coindex, result, *rank,
                             skip, count);
}

