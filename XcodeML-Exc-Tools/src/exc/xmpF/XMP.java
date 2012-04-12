/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import xcodeml.util.XmLog;

/**
 * all static members are defined here
 */
public class XMP {

  final static String prop = "XMPprop";
  public static boolean debugFlag = true;

  // defined in xmp_constant.h
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

  public final static int GMOVE_NORMAL	= 400;
  public final static int GMOVE_IN	= 401;
  public final static int GMOVE_OUT	= 402;

  public final static int MAX_DIM			= 7;
  public final static int NONBASIC_TYPE			= 599;

  public final static String PREFIX_			= "XMP__";
  public final static String DESC_PREFIX_		= "XMP_DESC_";

  public final static String ASTERISK			= "* @{ASTERISK}@";
  public final static String COLON			= ": @{COLON}@";

  public final static String nodes_alloc_f = "xmpf_nodes_alloc_";
  public final static String nodes_dim_size_f = "xmpf_nodes_dim_size_";
  public final static String nodes_init_GLOBAL_f = "xmpf_nodes_init_GLOBAL_";
  public final static String nodes_init_EXEC_f = "xmpf_nodes_init_EXEC_";
  public final static String nodes_init_NODES_f = "xmpf_nodes_init_NODES_";

  public final static String template_alloc_f = "xmpf_template_alloc_";
  public final static String template_dim_info_f = "xmpf_template_dim_info_";
  public final static String template_init_f = "xmpf_template_init_";

  public final static String array_alloc_f = "xmpf_array_alloc_";
  public final static String array_align_info_f = "xmpf_align_info_";
  public final static String array_init_f = "xmpf_array_init_";
  public final static String array_get_local_size_f = "xmpf_array_get_local_size_";
  public final static String array_set_local_array_f = "xmpf_array_set_local_array_";

  public final static String ref_templ_alloc_f = "xmpf_ref_templ_alloc_";
  public final static String ref_nodes_alloc_f = "xmpf_ref_nodes_alloc_";
  public final static String ref_set_info_f = "xmpf_ref_set_info_";
  public final static String ref_init_f = "xmpf_ref_init_";

  public final static String loop_test_f = "xmpf_loop_test_";
  
  public final static String arrayProp = "XMParrayProp";

  private static boolean errorFlag			= false;
  private static boolean errorFlags			= false;

  public static Xobject typeIntConstant(Xtype type) {
    if(type.isBasic())
      return Xcons.IntConstant(type.getBasicType() + 500);
    else 
      return Xcons.IntConstant(NONBASIC_TYPE);
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
    System.out.println("[XcalableMP] " + msg);
  }

  public static void error(LineNo l, String msg) {
    errorFlag = true;
    errorFlags = true;
    System.out.println("[XcalableMP:"+l+"] " + msg);
    XmLog.error(l, "[XcalableMP] " + msg);
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
