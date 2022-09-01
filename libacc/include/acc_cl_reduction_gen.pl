#
# OpenACC->OpenCL reduction macro generator
#

# _ACC_gpu_init_reduction_var_double_PLUS(&_ACC_reduction_bt_result);
# --> #define _ACC_gpu_init_reduction_var_double_PLUS(x)  _ACC_gpu_init_reduction_var_double(x, _ACC_REDUCTION_PLUS)
# inline
# void _ACC_gpu_init_reduction_var_double(double __global *var, int kind){
#   switch(kind){
#   case _ACC_REDUCTION_PLUS: *var = 0.0; return;
#   case _ACC_REDUCTION_MUL: *var = 1.0; return;
#   case _ACC_REDUCTION_MAX: *var = -DBL_MAX; return;
#   case _ACC_REDUCTION_MIN: *var = DBL_MAX; return;
#   }
# }

# _ACC_gpu_reduction_loc_set_double_PLUS(_ACC_reduction_loc_result,result,_ACC_GPU_RED_TMP);
#-> #define _ACC_gpu_reduction_loc_set_double_PLUS(reduction_loc,result_loc, reduction_tmp) \
#  if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
#  else reduction_loc = ((__global double *)reduction_tmp)+_ACC_block_idx_x;\
#  if(_ACC_thread_idx_x == 0) _ACC_gpu_init_reduction_Gvar_double(reduction_loc,_ACC_REDUCTION_PLUS); \
#  barrier(CLK_GLOBAL_MEM_FENCE); 

# _ACC_gpu_reduction_loc_update_double_PLUS(_ACC_reduction_loc_result,_ACC_reduction_bt_result);
#-> #define _ACC_gpu_reduction_loc_update_double_PLUS(reduction_loc, reduction_value) \
#    _ACC_reduce_threads_double(reduction_loc, reduction_value, _ACC_REDUCTION_PLUS, true)
#
# void _ACC_reduce_threads_double(volatile __global double *target, double input, int kind, bool do_acc)
# {
#     volatile union {
# 	real_t r;
# 	uint_t i;
#     } next, expected, current;

#     current.r = *target;
#     do {
# 	expected.r = current.r;
# 	next.r     = do_acc? op_double(expected.r,input,kind):input;
# 	current.i  = atom_cmpxchg((volatile uint_t __global*) target, expected.i, next.i);
#     } while (current.i != expected.i);
# }

# _ACC_gpu_reduction_loc_block_double_PLUS(result,_ACC_GPU_RED_TMP,0,_ACC_GPU_RED_NUM);
#-> #define _ACC_gpu_reduction_loc_block_double_PLUS(result_loc, reduction_tmp, off, reduction_tmp_size) \
# if((_ACC_thread_idx_x)==(0)) { \
#   double part_result; \
#   _ACC_gpu_init_reduction_var_double(&part_result, _ACC_REDUCTION_PLUS); \
#   for(int i = 0; i < reduction_tmp_size; i++) \
#       part_result = op_double(part_result, ((__global double *)reduction_tmp)[i], _ACC_REDUCTION_PLUS); \
#   *result_loc = op_double(*result_loc, part_result, _ACC_REDUCTION_PLUS); \
#   } 

#my @int_dtype_arr = ("char", "int","long");
my @int_dtype_arr = ("int","long");
my @float_dtype_arr = ('float','double');
#my @int_op_arr = ("PLUS","MUL","MAX","MIN","BITAND","BITOR","BITXOR","LOGAND","LOGOR");
my @int_op_arr = ("PLUS","MUL","MAX","MIN");
my @float_op_arr = ("PLUS","MUL","MAX","MIN");

#
# init_reduction_var
#
foreach my $dtype (@float_dtype_arr){
    foreach my $op (@float_op_arr){
	gen_init_reduction_var_float($dtype,$op);
    }
}

sub init_reduction_var {
    my($dtype,$op,$var) = @_;
    local($init_stmt);
    if(($dtype eq "float") || ($dtype eq "double")){
	if($op eq "PLUS") { $init_stmt ="*(".$var.")=0.0;" }
	elsif($op eq "MUL") { $init_stmt ="*(".$var.")=1.0;" }
	elsif($op eq "MAX") { $init_stmt = $dtype == "double" ? "*(".$var.")=-DBL_MAX;" : "*(".$var.")=-FLT_MAX;"; }
	elsif($op eq "MIN") { $init_stmt = $dtype == "double" ? "*(".$var.")=DBL_MAX;" : "*(".$var.")=FLT_MAX;"; }
    } else { 
	if($op eq "PLUS") { $init_stmt ="*(".$var.")=0;" }
	elsif($op eq "MUL") { $init_stmt ="*(".$var.")=1;" }
	elsif($op eq "MAX") { $init_stmt = $dtype == "long" ? "*(".$var.")=-LONG_MAX;" : "*(".$var.")=-INT_MAX;"; }
	elsif($op eq "MIN") { $init_stmt = $dtype == "long" ? "*(".$var.")=LONG_MAX;" : "*(".$var.")=INT_MAX;"; }
    }
    return $init_stmt;
}

sub gen_init_reduction_var_float {
    my($dtype,$op) = @_;
    local($init_stmt, $name);
    $init_stmt = &init_reduction_var($dtype,$op,"var");
    $name = "_ACC_gpu_init_reduction_var_".$dtype."_".$op;
    print "#define ".$name."(var) ".$init_stmt."\n";
}

foreach my $dtype (@int_dtype_arr){
    foreach my $op (@int_op_arr){
	gen_init_reduction_var_int($dtype,$op);
    }
}

sub gen_init_reduction_var_int {
    my($dtype,$op) = @_;
    local($init_stmt, $name);
    $init_stmt = &init_reduction_var($dtype,$op,"var");
    $name = "_ACC_gpu_init_reduction_var_".$dtype."_".$op;
    print "#define ".$name."(var) ".$init_stmt."\n";
}

#
# reduction_loc_set
#
foreach my $dtype (@float_dtype_arr){
    foreach my $op (@float_op_arr){
	gen_reduction_loc_set($dtype,$op);
    }
}

foreach my $dtype (@int_dtype_arr){
    foreach my $op (@int_op_arr){
	gen_reduction_loc_set($dtype,$op);
    }
}

sub gen_reduction_loc_set {
    my($dtype,$op) = @_;
    local($name);
    $name = "_ACC_gpu_reduction_loc_set_".$dtype."_".$op;
    print "#define ".$name."(reduction_loc,result_loc,reduction_tmp) { \\\n";
    print "if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \\\n";
    print "else reduction_loc = ((__global ".$dtype." *)reduction_tmp)+_ACC_block_idx_x;\\\n";
    print "if(_ACC_thread_idx_x == 0) ".&init_reduction_var($dtype,$op,"reduction_loc").";\\\n";
    print "barrier(CLK_GLOBAL_MEM_FENCE); }\n";
}

#
#  reduction_loc_block
#
foreach my $dtype (@float_dtype_arr){
    foreach my $op (@float_op_arr){
	gen_reduction_loc_block($dtype,$op);
    }
}

foreach my $dtype (@int_dtype_arr){
    foreach my $op (@int_op_arr){
	gen_reduction_loc_block($dtype,$op);
    }
}

sub reduction_op {
    my($dtype,$op,$left,$right) = @_;
    local($stmt);
    if($op eq "PLUS") { $stmt ="(".$left.")+(".$right.")" }
    elsif($op eq "MUL") { $stmt ="(".$left.")*(".$right.")" }
    elsif($op eq "MAX") { $stmt ="(".$left.")>(".$right.")?(".$left."):(".$right.")" }
    elsif($op eq "MIN") { $stmt ="(".$left.")<(".$right.")?(".$left."):(".$right.")" }
    return $stmt;
}

sub gen_reduction_loc_block {
    my($dtype,$op) = @_;
    local($name);
    $name = "_ACC_gpu_reduction_loc_block_".$dtype."_".$op;
    print "#define ".$name."(result_loc,reduction_tmp,off,tmp_size) {\\\n";
    print "if((_ACC_thread_idx_x)==(0)) { \\\n";
    print "double part_result; ".&init_reduction_var($dtype,$op,"&part_result").";\\\n";;
    print "for(int i = 0; i < tmp_size; i++) {\\\n";
    print $dtype." x="."((__global ".$dtype." *)reduction_tmp)[i];\\\n";
    print "part_result = ".&reduction_op($dtype,$op,"part_result","x")."; }\\\n";
    print "*result_loc = ".&reduction_op($dtype,$op, "*result_loc","part_result")."; }}\n";
}

#
#  reduction_loc_update
#
foreach my $dtype (@float_dtype_arr){
    foreach my $op (@float_op_arr){
	gen_reduction_loc_update($dtype,$op);
    }
}

foreach my $dtype (@int_dtype_arr){
    foreach my $op (@int_op_arr){
	gen_reduction_loc_update($dtype,$op);
    }
}

sub gen_reduction_loc_update {
    my($dtype,$op) = @_;
    local($name,$u_type);
    $name = "_ACC_gpu_reduction_loc_update_".$dtype."_".$op;
    print "#define ".$name."(target,input) {\\\n";
    if(($dtype eq "double")||($dtype eq "long")) { $u_type="ulong"; }
    else { $u_type = "uint"; }
    print "volatile union { ".$dtype." v; ".$u_type." i; } next, expected, current; \\\n";
    print "current.v = *target;\\\n";
    print "do { expected.i = current.i; \\\n";
    print "next.v=".&reduction_op($dtype,$op,"expected.v","input").";\\\n";
    print "current.i  = atom_cmpxchg((volatile ".$u_type." __global *) target, expected.i, next.i);";
    print "} while (current.i != expected.i);  }\n";
}
