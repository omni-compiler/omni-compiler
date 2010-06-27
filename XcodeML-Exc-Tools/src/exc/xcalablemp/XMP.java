package exc.xcalablemp;

import exc.object.*;
import xcodeml.util.XmLog;

/**
 * all static members are defined here
 */
public class XMP {
  public final static int MAX_DIM		= 7;
  public final static String DESC_PREFIX_	= "_XCALABLEMP_desc_";
  public final static String ADDR_PREFIX_	= "_XCALABLEMP_addr_";
  public final static String GTOL_PREFIX_	= "_XCALABLEMP_gtol_";
  public final static String ASTERISK		= "ASTERISK";
  public final static String COLON		= "COLON";

  private static boolean errorFlag		= false;

  public static void exitByError() {
    if (errorFlag) System.exit(1);
  }

  public static void error(LineNo l, String msg) throws XMPexception {
    errorFlag = true;
    XmLog.error(l, "[XcalableMP] " + msg);
    throw new XMPexception();
  }

  public static void warning(LineNo l, String msg) {
    XmLog.warning("[XcalableMP] " + msg);
  }

  public static void fatal(String msg) {
    XmLog.fatal("[XcalableMP] " + msg);
  }

  public static Ident getMacroId(String name) {
    return new Ident(name, StorageClass.EXTERN, Xtype.Function(Xtype.voidType),
                     Xcons.Symbol(Xcode.FUNC_ADDR, Xtype.Pointer(Xtype.Function(Xtype.voidType)), name), VarScope.GLOBAL);
  }
}
