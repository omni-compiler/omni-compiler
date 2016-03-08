/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import xcodeml.util.XmLog;

/**
 * all static members are defined here
 */
public class XMP {
  public static boolean debugFlag = false;

  final static String prop = "XMPprop";
  final static String RWprotected = "XMPRWprotected";

  // defined in xmp_constant.h
  public final static int REDUCE_NONE		= 0;
  public final static int REDUCE_SUM		= 300;
  public final static int REDUCE_PROD		= 301;
  public final static int REDUCE_BAND		= 302;
  public final static int REDUCE_LAND		= 303;
  public final static int REDUCE_BOR		= 304;
  public final static int REDUCE_LOR		= 305;
  public final static int REDUCE_BXOR		= 306;
  public final static int REDUCE_LXOR		= 307;
  public final static int REDUCE_MAX		= 308;
  public final static int REDUCE_MIN		= 309;
  public final static int REDUCE_FIRSTMAX	= 310;
  public final static int REDUCE_FIRSTMIN	= 311;
  public final static int REDUCE_LASTMAX	= 312;
  public final static int REDUCE_LASTMIN	= 313;
  public final static int REDUCE_EQV            = 314;
  public final static int REDUCE_NEQV           = 315;
  public final static int REDUCE_MINUS          = 316;
  public final static int REDUCE_MAXLOC         = 317;
  public final static int REDUCE_MINLOC         = 318;

  public final static int GMOVE_NORMAL	        = 400;
  public final static int GMOVE_IN              = 401;
  public final static int GMOVE_OUT	        = 402;

  public final static int MAX_DIM		= 7;
  public       static int MAX_ASSUMED_SHAPE     = 16;
  public final static int NONBASIC_TYPE		= 99 /*599*/;

  public final static String SIZE_ARRAY_NAME    = "xmp_size_array";
  public final static String XMP_COMMON_NAME    = "XMP_COMMON";

  public final static String PREFIX_		= "XMP__";
  public final static String DESC_PREFIX_	= "XMP_DESC_";
  public final static String SAVE_DESC_PREFIX_	= "XMP_SAVE_";    

  public final static String ASTERISK		= "* @{ASTERISK}@";
  public final static String COLON		= ": @{COLON}@";

  public final static String epilog_label_f     = "99999";

  public final static String nodes_alloc_f       = "xmpf_nodes_alloc_";
  public final static String nodes_dim_size_f    = "xmpf_nodes_dim_size_";
  public final static String nodes_init_GLOBAL_f = "xmpf_nodes_init_GLOBAL_";
  public final static String nodes_init_EXEC_f   = "xmpf_nodes_init_EXEC_";
  public final static String nodes_init_NODES_f  = "xmpf_nodes_init_NODES_";
  public final static String nodes_dealloc_f     = "xmpf_nodes_dealloc_";

  public final static String template_alloc_f    = "xmpf_template_alloc_";
  public final static String template_dim_info_f = "xmpf_template_dim_info_";
  public final static String template_init_f     = "xmpf_template_init_";
  public final static String template_dealloc_f  = "xmpf_template_dealloc_";

  public final static String array_alloc_f       = "xmpf_array_alloc_";
  public final static String init_allocated_f    = "xmp_f_init_allocated_";
  public final static String array_align_info_f  = "xmpf_align_info_";
  public final static String array_init_f        = "xmpf_array_init_";
  public final static String array_get_local_size_f  = "xmpf_array_get_local_size_off_";
  public final static String array_set_local_array_f = "xmpf_array_set_local_array_";
  public final static String array_init_shadow_f     = "xmpf_array_init_shadow_";
  public final static String array_dealloc_f         = "xmpf_array_dealloc_";
  public final static String array_deallocate_f      = "xmpf_array_deallocate_";

  public final static String ref_templ_alloc_f   = "xmpf_ref_templ_alloc_";
  public final static String ref_nodes_alloc_f   = "xmpf_ref_nodes_alloc_";
  public final static String ref_dealloc_f       = "xmpf_ref_dealloc_";
  public final static String ref_set_loop_info_f = "xmpf_ref_set_loop_info_";
  public final static String ref_set_dim_info_f  = "xmpf_ref_set_dim_info_";
  public final static String ref_init_f          = "xmpf_ref_init_";

  public final static String loop_test_f      = "xmpf_loop_test_";
  public final static String loop_test_skip_f = "xmpf_loop_test_skip_";
  public final static String loop_sched_f     = "xmpf_loop_sched_";
  public final static String l2g_f            = "xmpf_l2g_";

  public final static String set_reflect_f    = "xmpf_set_reflect_";
  public final static String reflect_f        = "xmpf_reflect_";
  public final static String reflect_async_f  = "xmpf_reflect_async_";
  public final static String init_async_f     = "xmpf_init_async_";
  public final static String start_async_f    = "xmpf_start_async_";
  public final static String wait_async_f     = "xmpf_wait_async_";
  public final static String barrier_f        = "xmpf_barrier_";
  public final static String reduction_f      = "xmpf_reduction_";
  public final static String reduction_loc_f  = "xmpf_reduction_loc_";
  public final static String bcast_f          = "xmpf_bcast_";

  public final static String create_task_nodes_f = "xmpf_create_task_nodes_";
  public final static String test_task_on_f      = "xmpf_test_task_on_nodes_";
  public final static String end_task_f          = "xmpf_end_task_";
  public final static String test_task_nocomm_f  = "xmpf_test_task_nocomm_";

  public final static String gmove_g_alloc_f     = "xmpf_gmv_g_alloc_";
  public final static String gmove_l_alloc_f     = "xmpf_gmv_l_alloc_";
  public final static String gmove_g_dim_info_f  = "xmpf_gmv_g_dim_info_";
  public final static String gmove_l_dim_info_f  = "xmpf_gmv_l_dim_info_";
  public final static String gmove_do_f          = "xmpf_gmv_do_";
  public final static String gmove_dealloc_f     = "xmpf_gmv_dealloc_";
  
  public final static String finalize_all_f = "xmpf_finalize_all_";

  public final static String arrayProp = "XMParrayProp";

  private static boolean errorFlag			= false;
  private static boolean errorFlags			= false;

  public static Xobject typeIntConstant(Xtype type) {
    if(type.isBasic()){
      return Xcons.IntConstant(reduceBasicType(type)+500);
    }else 
      return Xcons.IntConstant(NONBASIC_TYPE+500);
  }

  static int reduceBasicType(Xtype type){
    Xobject Fkind = null;
    int t = type.getBasicType();
    switch(t){
    case BasicType.INT:
      Fkind = type.getFkind();
      if(Fkind != null && Fkind.isIntConstant()){
	switch(Fkind.getInt()){
	case 1:
	  t = BasicType.CHAR;
	  break;
	case 2:
	  t = BasicType.SHORT;
	  break;
	case 4:
	  t = BasicType.INT;
	  break;
	case 8:
	  t = BasicType.LONGLONG;
	  break;
	}
      }
      break;
    case BasicType.BOOL:
      Fkind = type.getFkind();
      if(Fkind != null && Fkind.isIntConstant()){
	switch(Fkind.getInt()){
	case 1:
	  t = BasicType.UNSIGNED_CHAR;
	  break;
	case 2:
	  t = BasicType.UNSIGNED_SHORT;
	  break;
	case 4:
	  t = BasicType.UNSIGNED_LONG;
	  break;
	case 8:
	  t = BasicType.UNSIGNED_LONGLONG;
	  break;
	}
      }
      break;
    case BasicType.FLOAT:
      Fkind = type.getFkind();
      if(Fkind != null && Fkind.isIntConstant()){
	switch(Fkind.getInt()){
	case 4:
	  t = BasicType.FLOAT;
	  break;
	case 8:
	  t = BasicType.DOUBLE;
	  break;
	case 16:
	  t = BasicType.LONG_DOUBLE;
	  break;
	}
      }
      break;
    case BasicType.FLOAT_COMPLEX:
      Fkind = type.getFkind();
      if(Fkind != null && Fkind.isIntConstant()){
	switch(Fkind.getInt()){
	case 4:
	  t = BasicType.FLOAT_COMPLEX;
	  break;
	case 8:
	  t = BasicType.DOUBLE_COMPLEX;
	  break;
	case 16:
	  t = BasicType.LONG_DOUBLE_COMPLEX;
	  break;
	}
      }
      break;
    case BasicType.CHAR:
    case BasicType.UNSIGNED_CHAR:
    case BasicType.SHORT:
    case BasicType.UNSIGNED_SHORT:
    case BasicType.UNSIGNED_INT:
    case BasicType.LONG:
    case BasicType.UNSIGNED_LONG:
    case BasicType.LONGLONG:
    case BasicType.UNSIGNED_LONGLONG:
    case BasicType.DOUBLE:
    case BasicType.LONG_DOUBLE:
    case BasicType.FLOAT_IMAGINARY:
    case BasicType.DOUBLE_IMAGINARY:
    case BasicType.LONG_DOUBLE_IMAGINARY:
    case BasicType.DOUBLE_COMPLEX:
    case BasicType.LONG_DOUBLE_COMPLEX:
	  break;
    default:
      return NONBASIC_TYPE;
    }
    return t;
  }

  public static void exitByError() {
    if (errorFlag) System.exit(1);
  }

  /*
   * gensym
   */
  static int gensym_num = 0;
  
  public static String genSym(String prefix) {
    String newString = new String(prefix + String.valueOf(gensym_num));
    gensym_num++;
    return newString;
  }

  // FIXME no line number
  public static void error(String msg) {
    errorFlag = true;
    errorFlags = true;
    System.err.println("[XcalableMP] " + msg);
  }

  public static void errorAt(Block b, String msg) {
    LineNo l = null;

    if(b == null || (l = b.getLineNo()) == null){
      error(msg);
      return;
    }
      
    errorFlag = true;
    errorFlags = true;
    System.err.println("\""+l.fileName()+"\", line "+l.lineNo()+":(XMP) "+msg);
    // XmLog.error(l, "[XcalableMP] " + msg);
  }

  public static void resetError(){
    errorFlag = false;
  }

  public static void warning(String msg) {
    XmLog.warning("[XcalableMP] " + msg);
  }

  public static void fatal(String msg) {
    XmLog.fatal("[XcalableMP fatal] " + msg);
  }

  public static void fatal(Block b, String msg) {
    LineNo l = null;

    if (b == null || (l = b.getLineNo()) == null){
      XmLog.fatal(msg);
    }
      
    errorFlag = true;
    errorFlags = true;
    XmLog.fatal("\""+l.fileName() + "\", line " + l.lineNo() + ":(XMP) " + msg);
  }

  public static boolean hasError(){
    return errorFlag;
  }

  public static boolean hasErrors(){
    return errorFlags;
  }

  public static void debug(String msg){
    if(debugFlag)  System.out.println(msg);
  }
}
