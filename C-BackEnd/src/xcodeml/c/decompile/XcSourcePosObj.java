package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.util.XmcWriter;
import xcodeml.util.XmOption;
import xcodeml.util.XmStringUtil;

/**
 * Internal object represents following elements:
 *   attribute of file, lineno
 */
public final class XcSourcePosObj implements XcAppendable
{
    /* source file path */
    private String _filePath;

    /* line number of original source code.*/
    private int _lineNum;

    /* line number of input file*/
    private int _rawLineNum;

    /**
     * Creates a XcSourcePosObj.
     * 
     * @param filePath the file path of the source code.
     * @param lineNum the line number in the source code.
     * @param rawLineNum the raw line number in the source code.
     */
    public XcSourcePosObj(String filePath,
                          int lineNum,
                          int rawLineNum)
    {
        _filePath = XmStringUtil.trim(filePath);
        _lineNum = lineNum;
        _rawLineNum = rawLineNum;
    }

    /**
     * Creates a XcSourcePosObj.
     * 
     * @param filePath the file path of the source code.
     * @param lineNumStr the line number in the source code.
     * @param rawLineNumStr the raw line number in the source code.
     */
    public XcSourcePosObj(String filePath,
                          String lineNumStr,
                          String rawLineNumStr)
    {
        _filePath = XmStringUtil.trim(filePath);

        if(lineNumStr != null) {
            try {
                _lineNum = Integer.parseInt(lineNumStr);
            } catch(NumberFormatException e) {
                throw new IllegalArgumentException("invalid line number");
            }
        }

        if(rawLineNumStr != null) {
            try {
                _rawLineNum = Integer.parseInt(rawLineNumStr);
            } catch(NumberFormatException e) {
                throw new IllegalArgumentException("invalid raw line number");
            }
        }
    }

    /**
     * Gets the file path of the source code.
     * 
     * @return the file path of the source code.
     */
    public final String getFilePath()
    {
        return _filePath;
    }

    /**
     * Gets the line number in the source code.
     * 
     * @return the line number in the source code.
     */
    public final int getLineNum()
    {
        return _lineNum;
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        if(XmOption.isSuppressLineDirective())
            return;

        if(_lineNum <= 0 || _filePath == null)
            return;

        w.noLfOrLf().add("# ").add(_lineNum);

        if(_filePath != null) {
            w.add(" \"").add(_filePath);

            if(w.getIsDebugMode() &&
               _rawLineNum > 0)
                w.addSpc("(raw line = ").addSpc(_rawLineNum).add(")");

            w.add("\"");
        }

        w.lf();
    }
}
