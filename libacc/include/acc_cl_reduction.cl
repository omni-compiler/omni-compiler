#define _ACC_gpu_init_reduction_var_float_PLUS(var) *(var)=0.0;
#define _ACC_gpu_init_reduction_var_float_MUL(var) *(var)=1.0;
#define _ACC_gpu_init_reduction_var_float_MAX(var) *(var)=-DBL_MAX;
#define _ACC_gpu_init_reduction_var_float_MIN(var) *(var)=DBL_MAX;
#define _ACC_gpu_init_reduction_var_double_PLUS(var) *(var)=0.0;
#define _ACC_gpu_init_reduction_var_double_MUL(var) *(var)=1.0;
#define _ACC_gpu_init_reduction_var_double_MAX(var) *(var)=-DBL_MAX;
#define _ACC_gpu_init_reduction_var_double_MIN(var) *(var)=DBL_MAX;
#define _ACC_gpu_init_reduction_var_int_PLUS(var) *(var)=0;
#define _ACC_gpu_init_reduction_var_int_MUL(var) *(var)=1;
#define _ACC_gpu_init_reduction_var_int_MAX(var) *(var)=-LONG_MAX;
#define _ACC_gpu_init_reduction_var_int_MIN(var) *(var)=LONG_MAX;
#define _ACC_gpu_init_reduction_var_long_PLUS(var) *(var)=0;
#define _ACC_gpu_init_reduction_var_long_MUL(var) *(var)=1;
#define _ACC_gpu_init_reduction_var_long_MAX(var) *(var)=-LONG_MAX;
#define _ACC_gpu_init_reduction_var_long_MIN(var) *(var)=LONG_MAX;
#define _ACC_gpu_reduction_loc_set_float_PLUS(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global float *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=0.0;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_float_MUL(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global float *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=1.0;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_float_MAX(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global float *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=-DBL_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_float_MIN(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global float *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=DBL_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_double_PLUS(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global double *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=0.0;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_double_MUL(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global double *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=1.0;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_double_MAX(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global double *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=-DBL_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_double_MIN(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global double *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=DBL_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_int_PLUS(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global int *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=0;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_int_MUL(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global int *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=1;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_int_MAX(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global int *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=-LONG_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_int_MIN(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global int *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=LONG_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_long_PLUS(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global long *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=0;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_long_MUL(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global long *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=1;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_long_MAX(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global long *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=-LONG_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_set_long_MIN(reduction_loc,result_loc,reduction_tmp) { \
if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
else reduction_loc = ((__global long *)reduction_tmp)+_ACC_block_idx_x;\
if(_ACC_thread_idx_x == 0) *(reduction_loc)=LONG_MAX;;\
barrier(CLK_GLOBAL_MEM_FENCE); }
#define _ACC_gpu_reduction_loc_block_float_PLUS(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=0.0;;\
for(int i = 0; i < tmp_size; i++) {\
float x=((__global float *)reduction_tmp)[i];\
part_result = (part_result)+(x); }\
*result_loc = (*result_loc)+(part_result); }}
#define _ACC_gpu_reduction_loc_block_float_MUL(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=1.0;;\
for(int i = 0; i < tmp_size; i++) {\
float x=((__global float *)reduction_tmp)[i];\
part_result = (part_result)*(x); }\
*result_loc = (*result_loc)*(part_result); }}
#define _ACC_gpu_reduction_loc_block_float_MAX(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=-DBL_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
float x=((__global float *)reduction_tmp)[i];\
part_result = (part_result)>(x)?(part_result):(x); }\
*result_loc = (*result_loc)>(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_block_float_MIN(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=DBL_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
float x=((__global float *)reduction_tmp)[i];\
part_result = (part_result)<(x)?(part_result):(x); }\
*result_loc = (*result_loc)<(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_block_double_PLUS(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=0.0;;\
for(int i = 0; i < tmp_size; i++) {\
double x=((__global double *)reduction_tmp)[i];\
part_result = (part_result)+(x); }\
*result_loc = (*result_loc)+(part_result); }}
#define _ACC_gpu_reduction_loc_block_double_MUL(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=1.0;;\
for(int i = 0; i < tmp_size; i++) {\
double x=((__global double *)reduction_tmp)[i];\
part_result = (part_result)*(x); }\
*result_loc = (*result_loc)*(part_result); }}
#define _ACC_gpu_reduction_loc_block_double_MAX(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=-DBL_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
double x=((__global double *)reduction_tmp)[i];\
part_result = (part_result)>(x)?(part_result):(x); }\
*result_loc = (*result_loc)>(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_block_double_MIN(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=DBL_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
double x=((__global double *)reduction_tmp)[i];\
part_result = (part_result)<(x)?(part_result):(x); }\
*result_loc = (*result_loc)<(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_block_int_PLUS(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=0;;\
for(int i = 0; i < tmp_size; i++) {\
int x=((__global int *)reduction_tmp)[i];\
part_result = (part_result)+(x); }\
*result_loc = (*result_loc)+(part_result); }}
#define _ACC_gpu_reduction_loc_block_int_MUL(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=1;;\
for(int i = 0; i < tmp_size; i++) {\
int x=((__global int *)reduction_tmp)[i];\
part_result = (part_result)*(x); }\
*result_loc = (*result_loc)*(part_result); }}
#define _ACC_gpu_reduction_loc_block_int_MAX(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=-LONG_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
int x=((__global int *)reduction_tmp)[i];\
part_result = (part_result)>(x)?(part_result):(x); }\
*result_loc = (*result_loc)>(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_block_int_MIN(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=LONG_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
int x=((__global int *)reduction_tmp)[i];\
part_result = (part_result)<(x)?(part_result):(x); }\
*result_loc = (*result_loc)<(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_block_long_PLUS(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=0;;\
for(int i = 0; i < tmp_size; i++) {\
long x=((__global long *)reduction_tmp)[i];\
part_result = (part_result)+(x); }\
*result_loc = (*result_loc)+(part_result); }}
#define _ACC_gpu_reduction_loc_block_long_MUL(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=1;;\
for(int i = 0; i < tmp_size; i++) {\
long x=((__global long *)reduction_tmp)[i];\
part_result = (part_result)*(x); }\
*result_loc = (*result_loc)*(part_result); }}
#define _ACC_gpu_reduction_loc_block_long_MAX(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=-LONG_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
long x=((__global long *)reduction_tmp)[i];\
part_result = (part_result)>(x)?(part_result):(x); }\
*result_loc = (*result_loc)>(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_block_long_MIN(result_loc,reduction_tmp,off,tmp_size) {\
if((_ACC_thread_idx_x)==(0)) { \
double part_result; *(&part_result)=LONG_MAX;;\
for(int i = 0; i < tmp_size; i++) {\
long x=((__global long *)reduction_tmp)[i];\
part_result = (part_result)<(x)?(part_result):(x); }\
*result_loc = (*result_loc)<(part_result)?(*result_loc):(part_result); }}
#define _ACC_gpu_reduction_loc_update_float_PLUS(target,input) {\
volatile union { float v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)+(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_float_MUL(target,input) {\
volatile union { float v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)*(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_float_MAX(target,input) {\
volatile union { float v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)>(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_float_MIN(target,input) {\
volatile union { float v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)<(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_double_PLUS(target,input) {\
volatile union { double v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)+(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_double_MUL(target,input) {\
volatile union { double v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)*(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_double_MAX(target,input) {\
volatile union { double v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)>(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_double_MIN(target,input) {\
volatile union { double v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)<(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_int_PLUS(target,input) {\
volatile union { int v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)+(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_int_MUL(target,input) {\
volatile union { int v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)*(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_int_MAX(target,input) {\
volatile union { int v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)>(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_int_MIN(target,input) {\
volatile union { int v; uint i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)<(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile uint __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_long_PLUS(target,input) {\
volatile union { long v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)+(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_long_MUL(target,input) {\
volatile union { long v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)*(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_long_MAX(target,input) {\
volatile union { long v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)>(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
#define _ACC_gpu_reduction_loc_update_long_MIN(target,input) {\
volatile union { long v; ulong i; } next, expected, current; \
current.v = *target;\
do { expected.i = current.i; \
next.v=(expected.v)<(input)?(expected.v):(input);\
current.i  = atom_cmpxchg((volatile ulong __global *) target, expected.i, next.i);} while (current.i != expected.i);  }
