#ifndef _ACC_GPU_ATOMIC
#define _ACC_GPU_ATOMIC


//float atomic func
__device__ static float atomicMul(float* address, float val)
{
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(__int_as_float(assumed) * val));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    if(__int_as_float(assumed) >= val) break;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    if(__int_as_float(assumed) <= val) break;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}



//double atomic func
/* this code was taken from CUDA C programming guide */
__device__ static double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ static double atomicMul(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ static double atomicMax(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    if(__longlong_as_double(assumed) >= val) break;
    old = atomicCAS(address_as_ull, assumed, val);
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ static double atomicMin(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    if(__longlong_as_double(assumed) <= val) break;
    old = atomicCAS(address_as_ull, assumed, val);
  } while (assumed != old);
  return __longlong_as_double(old);
}


#endif //_ACC_GPU_ATOMIC
