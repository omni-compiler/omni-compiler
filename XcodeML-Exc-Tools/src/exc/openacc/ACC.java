package exc.openacc;

import exc.object.*;
import xcodeml.util.XmLog;

/**
 * all static members are defined here
 */
public class ACC {
  public final static String prop = "_ACC_PROP_";
  
  private static boolean errorFlag      = false;
  
  public static boolean useReadOnlyDataCache = true;

  public static final int ACC_ASYNC_SYNC = -1;
  public static final int ACC_ASYNC_NOVAL = -2;
  public static final int HOST_TO_DEVICE = 400;
  public static final int DEVICE_TO_HOST = 401;

  public static final String INIT_DATA_FUNC_NAME = "_ACC_init_data";
  public static final String PRESENT_OR_INIT_DATA_FUNC_NAME = "_ACC_pinit_data";
  public static final String FINALIZE_DATA_FUNC_NAME = "_ACC_finalize_data";
  public static final String COPY_DATA_FUNC_NAME = "_ACC_copy_data";
  public static final String COPY_SUBDATA_FUNC_NAME = "_ACC_copy_subdata";
  public static final String PRESENT_OR_COPY_DATA_FUNC_NAME = "_ACC_pcopy_data";
  public static final String FIND_DATA_FUNC_NAME = "_ACC_find_data";


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

  public static void warning(LineNo l, String msg){
    XmLog.warning(l, "[OpenACC] " + msg);
  }

  public static void fatal(String msg) {
    XmLog.fatal("[OpenACC] " + msg);
  }
  
  public static void debug(String msg) {
    XmLog.debug("[OpenACC] " + msg);
  }
}
