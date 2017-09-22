/*
 *   COARRAY PUT
 *
 */

#include "xmpf_internal_coarray.h"

/***************************************************\
    entry
\***************************************************/

void xmpf_coarray_put_scalar_(void **descPtr, char **baseAddr, int *element,
                              int *coindex, char *rhs, BOOL *synchronous)
{
  XMPCO_PUT_scalarStmt(*descPtr, *baseAddr, *element,
                       *coindex, rhs, *synchronous);
}


void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                             int *coindex, char **rhsAddr, int *rank,
                             int skip[], int skip_rhs[], int count[],
                             BOOL *synchronous)
{
  XMPCO_PUT_arrayStmt(*descPtr, *baseAddr, *element,
                      *coindex, *rhsAddr, *rank,
                      skip, skip_rhs, count,
                      *synchronous);
}


void xmpf_coarray_put_spread_(void **descPtr, char **baseAddr, int *element,
                              int *coindex, char *rhs, int *rank,
                              int skip[], int count[], BOOL *synchronous)
{
  XMPCO_PUT_spread(*descPtr, *baseAddr, *element,
                   *coindex, rhs, *rank,
                   skip, count, *synchronous);
}


