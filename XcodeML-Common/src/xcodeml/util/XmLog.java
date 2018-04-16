package xcodeml.util;

import xcodeml.util.ILineNo;
import xcodeml.util.IXobject;

public class XmLog
{
    private static void printlnToOut(String s)
    {
        System.out.println(s);
        System.out.flush();
    }
    
    private static void printlnToErr(String s)
    {
        System.err.println(s);
        System.err.flush();
    }
    
    public static void info(String s)
    {
        printlnToOut(s);
    }
    
    public static void warning(String s)
    {
        printlnToErr("warn: " + s);
    }
    
    public static void warning(ILineNo lineNo, String s)
    {
        warning(wrapLocation(lineNo, s));
    }
    
    public static void error(String s)
    {
        printlnToErr("error: " + s);
    }
    
    public static void error(ILineNo lineNo, String s)
    {
        error(wrapLocation(lineNo, s));
    }
    
    private static String getExceptionMessage(Exception e)
    {
        if(e.getCause() != null)
            return e.getCause().getMessage();
        return e.getMessage();
    }
    
    public static void error(ILineNo lineNo, Exception e)
    {
        error(wrapLocation(lineNo, getExceptionMessage(e)));
        if(XmOption.isDebugOutput())
            e.printStackTrace();
    }
    
    // public static void error(XmObj xmobj, String s)
    // {
    //     error(wrapLocation(xmobj, s));
    // }
    
    // public static void error(XmObj xmobj, Exception e)
    // {
    //     error(wrapLocation(xmobj, getExceptionMessage(e)));
    //     if(XmOption.isDebugOutput())
    //         e.printStackTrace();
    // }
    
    public static void fatal(String s)
    {
        System.out.flush();
        System.err.flush();
        printlnToErr("fatal: " + s);
        Thread.dumpStack();
        System.exit(1);
    }
    
    public static void fatal_dump(String s, IXobject x)
    {
        System.out.flush();
        System.err.flush();
        printlnToErr("fatal: " + s);
        printlnToErr("Xcode = " + x.toString());
        Thread.dumpStack();
        System.exit(1);
    }
    
    public static void fatal(Throwable e)
    {
        printlnToErr("fatal:");
        e.printStackTrace();
        Thread.dumpStack();
    }
    
    // public static void fatal(XmObj xmobj, String s)
    // {
    //     fatal(wrapLocation(xmobj, s));
    // }
    
    public static void fatal(ILineNo lineNo, String s)
    {
        fatal(wrapLocation(lineNo, s));
    }
    
    public static void debug(String s)
    {
        if(XmOption.isDebugOutput())
            printlnToOut(s);
    }
    
    public static void debugAlways(String s)
    {
        printlnToOut(s);
    }
    
    // private static String getLeadText(XmObj xmobj)
    // {
    //     if(xmobj == null)
    //         return "";
        
    //     StringBuffer buf = new StringBuffer();
    //     xmobj.makeTextElement(buf);
    //     String s = buf.toString();
    //     buf = null;
    //     final int maxlen = 20;
    //     int len = (s.length() < maxlen) ? s.length() : maxlen;
    //     return s.substring(0, len);
    // }
    
    // private static String wrapLocation(XmObj xmobj, String s)
    // {
    //     if(xmobj instanceof IXbILineNo) {
    //         IXbLineNo lineno = (IXbLineNo)xmobj;
    //         return ((lineno.getFile() != null ? lineno.getFile() : "") + ":" +
    //             lineno.getLineno() + ": " + s);
    //     } else {
    //         return (s + "at " + getLeadText(xmobj));
    //     }
    // }

    private static String wrapLocation(ILineNo lineno, String s)
    {
        if(lineno == null)
            return s;
        return ((lineno.fileName() != null) ? lineno.fileName() : "") + ":" +
            lineno.lineNo() + ": " + s;
    }
}
