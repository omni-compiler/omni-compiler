package exc.openacc;

import exc.object.*;
import xcodeml.util.XmLog;

/**
 * all static members are defined here
 */
public class ACC {
  public final static String prop = "_ACC_PROP_";
  
  public static boolean debugFlag = true;
  private static boolean errorFlag      = false;

  public static Xobject createBasicTypeConstantObj(Xtype type) {
    return Xcons.IntConstant(type.getBasicType() + 500);
  }

  public static void exitByError() {
    if (errorFlag) System.exit(1);
  }

  public static void error(LineNo l, String msg) {
    errorFlag = true;
    XmLog.error(l, "[OpenACC] " + msg);
  }

  public static void warning(String msg) {
    XmLog.warning("[OpenACC] " + msg);
  }

  public static void fatal(String msg) {
    XmLog.fatal("[OpenACC] " + msg);
  }
  
  public static void debug(String msg) {
    XmLog.debug("[OpenACC] " + msg);
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
