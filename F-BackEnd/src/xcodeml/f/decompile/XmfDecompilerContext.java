/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import xcodeml.f.util.XmfWriter;
import xcodeml.util.XmDecompilerContext;
import xcodeml.util.XmOption;

/**
 * Decompile context.
 */
public class XmfDecompilerContext implements XmDecompilerContext
{
    private int _maxColumnCount = XmfWriter.DEFAULT_MAX_COLUMN_COUNT;
    private String _lastErrorMessage = null;

    private XmfWriter  _writer;
    //private XfTypeManager _typeManager;
    private XfTypeManagerForDom _typeManagerForDom;

    private Exception _lastCause;

    public XmfDecompilerContext()
    {
        //_typeManager = new XfTypeManager();
        _typeManagerForDom = new XfTypeManagerForDom();
    }

    @Override
    public void setProperty(String key, Object value)
    {
        if(key.equals(KEY_MAX_COLUMNS)) {
            setMaxColumnCount(Integer.parseInt(value.toString()));
        }
    }

    /**
     * Judge debug mode.
     * @return true/false.
     */
    public boolean isDebugMode()
    {
        return XmOption.isDebugOutput();
    }
    
    /**
     * Return cause of last error.
     */
    public Exception getLastCause()
    {
        return _lastCause;
    }

    /**
     * Judge output line directive.
     * @return true/false.
     */
    public boolean isOutputLineDirective()
    {
        return !XmOption.isSuppressLineDirective();
    }

    /**
     * Set output line directive flag.
     * @param outputLineDirective Output line directive flag.
     */
    public void setOutputLineDirective(boolean outputLineDirective)
    {
        XmOption.setIsSuppressLineDirective(!outputLineDirective);
    }

    /**
     * Get mumber of max columns.
     * @return Number of max columns.
     */
    public int getMaxColumnCount()
    {
        return _maxColumnCount;
    }

    /**
     * Set mumber of max columns.
     * @param maxColumnCount Number of max columns.
     */
    public void setMaxColumnCount(int maxColumnCount)
    {
        _maxColumnCount = maxColumnCount;
    }

    /**
     * Get writer for Fortran.
     * @return Instance of XmfWriter.
     */
     public XmfWriter getWriter()
    {
        return _writer;
    }

    /**
     * Set writer for Fortran.
     * @param writer XfTypeManager.
     */
    public void setWriter(XmfWriter writer)
    {
        _writer = writer;
    }

    // /**
    //  * Get type manager.
    //  * @return Instance of XfTypeManager.
    //  */
    // public XfTypeManager getTypeManager()
    // {
    //     return _typeManager;
    // }

    /**
     * Get type manager.
     * @return Instance of XfTypeManager.
     */
    public XfTypeManagerForDom getTypeManagerForDom()
    {
        return _typeManagerForDom;
    }

    public boolean hasError()
    {
        if (_lastErrorMessage != null) {
            return true;
        }
        return false;
    }

    public void setLastErrorMessage(String message)
    {
        _lastErrorMessage = message;
        _lastCause = new Exception();
    }

    public String getLastErrorMessage()
    {
        if (hasError() == false) {
            return XfError.SUCCESS.message();
        }
        return _lastErrorMessage;
    }

    public void debugPrint(String message)
    {
        if (isDebugMode() != false) {
            System.out.print("Debug: ");
            System.out.print(message);
        }
    }

    public void debugPrint(String format, Object... args)
    {
        if (isDebugMode() != false) {
            System.out.print("Debug: ");
            System.out.format(format, args);
        }
    }

    public void debugPrintLine(String message)
    {
        if (isDebugMode() != false) {
            System.out.print("Debug: ");
            System.out.println(message);
        }
    }
}
