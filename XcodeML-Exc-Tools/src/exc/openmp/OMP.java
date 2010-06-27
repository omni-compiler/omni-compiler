/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import xcodeml.util.XmLog;
import exc.object.*;

/**
 * for OMP, all static member defined here
 */
public class OMP
{
    final static String prop = "OMPprop";
    public static boolean debugFlag = false;

    final static String thdprv_prop = "ThreadPrivate";

    /*
     * mode control variables in this OMP package
     */
    static boolean leaveThreadPrivateFlag;
    static boolean moveAutoFlag;

    private static boolean errorFlag;
    private static boolean errorFlags;
    
    static public void setMoveAutoFlag(boolean f)
    {
        moveAutoFlag = f;
    }

    static public void leaveThreadPrivate(boolean f)
    {
        leaveThreadPrivateFlag = f;
    }

    public static void setThreadPrivate(Ident id)
    {
        id.setProp(thdprv_prop, thdprv_prop);
    }

    public static boolean isThreadPrivate(Ident id)
    {
        if(id == null)
            return false;
        return (id.getProp(thdprv_prop) != null);
    }
    
    static boolean hasError()
    {
        return errorFlag;
    }
    
    public static boolean hasErrors()
    {
        return errorFlags;
    }
    
    public static void resetError()
    {
        errorFlag = false;
    }

    public static void error(LineNo l, String msg)
    {
        errorFlag = true;
        errorFlags = true;
        XmLog.error(l, "[OpenMP] " + msg);
    }

    public static void warning(LineNo l, String msg)
    {
        XmLog.warning("[OpenMP] " + msg);
    }

    public static void fatal(String msg)
    {
        XmLog.fatal("[OpenMP] " + msg);
    }
    
    public static void debug(String msg)
    {
        if(debugFlag)
            XmLog.debugAlways(msg);
    }
}
