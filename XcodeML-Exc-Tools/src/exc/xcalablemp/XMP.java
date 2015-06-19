/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import xcodeml.util.XmLog;

/**
 * all static members are defined here
 */
public class XMP {
  public final static int MAX_DIM			= 7;
  public final static int NONBASIC_TYPE			= 599;
  public final static String TEMP_PREFIX		= "_XMP_TEMP";
  public final static String DESC_PREFIX_		= "_XMP_DESC_";
  public final static String NODES_SIZE_PREFIX_		= "_XMP_NODES_SIZE_";
  public final static String NODES_RANK_PREFIX_		= "_XMP_NODES_RANK_";
  public final static String STATIC_DESC_PREFIX_        = "_XMP_STATIC_";
  public final static String GPU_HOST_DESC_PREFIX_	= "_XMP_GPU_HOST_DESC_";
  public final static String GPU_DEVICE_DESC_PREFIX_	= "_XMP_GPU_DEVICE_DESC_";
  public final static String GPU_DEVICE_ADDR_PREFIX_	= "_XMP_GPU_DEVICE_ADDR_";
  public final static String GPU_FUNC_PREFIX		= "_XMP_GPU_FUNC";
  public final static String COARRAY_DESC_PREFIX_	= "_XMP_COARRAY_DESC_";
  public final static String COARRAY_ADDR_PREFIX_       = "_XMP_COARRAY_ADDR_";
  public final static String ADDR_PREFIX_		= "_XMP_ADDR_";
  public final static String GTOL_PREFIX_		= "_XMP_GTOL_";
  public final static String ASTERISK			= "* @{ASTERISK}@";
  public final static String COLON			= ": @{COLON}@";

  private static boolean errorFlag			= false;

  public static Xobject createBasicTypeConstantObj(Xtype type) {
    return Xcons.IntConstant(type.getBasicType() + 500);
  }

  public static void exitByError() {
    if (errorFlag) System.exit(1);
  }

  public static void error(LineNo l, String msg) {
    errorFlag = true;
    XmLog.error(l, "[XcalableMP] " + msg);
  }

  public static void warning(String msg) {
    XmLog.warning("[XcalableMP] " + msg);
  }

  public static void fatal(String msg) {
    XmLog.fatal("[XcalableMP] " + msg);
  }

  public static void fatal(LineNo lineNo, String msg) {
    XmLog.fatal("[XcalableMP] " + lineNo + " : " + msg);
  }

  public static Ident getMacroId(String name) {
    return new Ident(name, StorageClass.EXTERN, Xtype.Function(Xtype.voidType),
                     Xcons.Symbol(Xcode.FUNC_ADDR, Xtype.Pointer(Xtype.Function(Xtype.voidType)), name), VarScope.GLOBAL);
  }

  public static Ident getMacroId(String name, Xtype type) {
    return new Ident(name, StorageClass.EXTERN, Xtype.Function(type),
                     Xcons.Symbol(Xcode.FUNC_ADDR, Xtype.Pointer(Xtype.Function(type)), name), VarScope.GLOBAL);
  }
}
