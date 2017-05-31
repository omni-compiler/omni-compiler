/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.util;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.Writer;

import xcodeml.util.XmException;
import xcodeml.c.decompile.XcAppendable;
import xcodeml.c.decompile.XcOperatorEnum;
import xcodeml.c.decompile.XcOperatorObj;
import xcodeml.c.decompile.XcCompoundValueObj;
import xcodeml.c.type.XcIdent;

/**
 * Some functional writer.
 */
public class XmcWriter
{
    private PrintWriter _writer;

    private StringWriter _swriter;

    private char _lastChar;

    private boolean _isDebugMode = false;

    /**
     * Creates XmWriter.
     */
    public XmcWriter()
    {
        this(4 * 1024);
    }

    /**
     * Creates XmWriter.
     *
     * @param initCapacity initial capacity.
     */
    public XmcWriter(int initCapacity)
    {
        _swriter = new StringWriter(initCapacity);
        _writer = new PrintWriter(_swriter);
    }

    /**
     * Creates XmWriter.
     *
     * @param writer
     */
    public XmcWriter(Writer writer)
    {
        _writer = new PrintWriter(writer);
    }

    /**
     * Sets debug mode of XmWriter.
     *
     * @param enable enable of debug mode
     */
    public void setIsDebugMode(boolean enable)
    {
        _isDebugMode = enable;
    }

    /**
     * Tests writer enable debug mode.
     *
     * @return true if debug mode is enabled
     */
    public boolean getIsDebugMode()
    {
        return _isDebugMode;
    }

    private void _print(String s)
    {
        if(s == null || s.length() == 0)
            return;
        _lastChar = s.charAt(s.length() - 1);
        _writer.print(s);
    }
    
    private void _print(char c)
    {
        if(c == 0)
            return;
        _lastChar = c;
        _writer.print(c);
    }
   
    private void _print(int n)
    {
        _lastChar = '0';
        _writer.print(n);
    }
    
    private void _print(double n)
    {
        _lastChar = '0';
        _writer.print(n);
    }

    /**
     * Outputs a string with new line if string is not null and zero length.
     *
     * @param s output string
     */
    public void println(String s)
    {
        _println(s);
    }

    private void _println(String s)
    {
        _lastChar = '\n';
        if(s == null || s.length() == 0)
            return;
        _writer.println(s);
    }

    private void _println()
    {
        _lastChar = '\n';
        _writer.println("");
    }

    /**
     * Adds space if needed.
     *
     * @return this writer
     */
    public final XmcWriter spc()
    {
        if(_lastChar != 0) {
            switch(_lastChar)
            {
            case ' ': case '\n' : case '\t': case '[': case '(': case '{':
                break;
            default:
                _print(" ");
            }
        }
        return this;
    }

    /**
     * Adds LF.
     *
     * @return this writer
     */
    public final XmcWriter lf()
    {
        _println();
        return this;
    }

    /**
     * Adds LF if needed.
     *
     * @return this writer
     */
    public final XmcWriter noLfOrLf()
    {
        if(_lastChar != '\n')
            _println();
        return this;
    }

    /**
     * Adds ';', LF
     * (end of statement)
     *
     * @return this writer
     */
    public final XmcWriter eos()
    {
        _println(";");
        return this;
    }

    /**
     * Adds string with no LF
     *
     * @param s string
     * @return this writer
     */
    public final XmcWriter add(String s)
    {
        if(s != null)
            _print(s);
        return this;
    }
   
    /**
     * Adds char with no LF
     *
     * @param c char
     * @return this writer
     */
    public final XmcWriter add(char c)
    {
        _print(c);
        return this;
    }

    /**
     * Adds string with space if needed
     *
     * @param s string
     * @return this writer
     */
    public final XmcWriter addSpc(String s)
    {
        spc();
        if(s != null)
            _print(s);
        return this;
    }

    /**
     * Adds integer with no LF
     *
     * @param n
     * @return this writer
     */
    public final XmcWriter add(int n)
    {
        _print(n);
        return this;
    }

    /**
     * Adds double with no LF
     *
     * @param n
     * @return this writer
     */
    public final XmcWriter add(double n)
    {
        _print(n);
        return this;
    }

    /**
     * Adds integer with space if needed
     *
     * @param n
     * @return this writer
     */
    public final XmcWriter addSpc(int n)
    {
        spc();
        _print(n);
        return this;
    }

    /**
     * Adds a object with brace
     *
     * @param obj a object added with brace
     * @return this writer
     */
    public final XmcWriter addBrace(XcAppendable obj) throws XmException
    {
        if(obj == null)
            return this;

        if(obj != null) {
            _print('(');
            obj.appendCode(this);
            _print(')');
        }
        return this;
    }

    /**
     * Adds a object with brace if needed (not brace identifier, compound value, comma expression)
     *
     * @param obj a object added with brace
     * @return this writer
     */
    public final XmcWriter addBraceIfNeeded(XcAppendable obj) throws XmException
    {
        if(obj != null) {
            if(obj instanceof XcIdent ||
               obj instanceof XcCompoundValueObj ||
                ((obj instanceof XcOperatorObj) && ((XcOperatorObj)obj).getOperatorEnum() == XcOperatorEnum.COMMA)) {
                obj.appendCode(this);
            } else {
                _print('(');
                obj.appendCode(this);
                _print(')');
            }
        }
        return this;
    }

    /**
     * Adds space, object
     *
     * @param obj a object added
     * @return this writer
     */
    public final XmcWriter addSpc(XcAppendable obj) throws XmException
    {
        spc();
        return add(obj);
    }
    
    /**
     * Adds object
     *
     * @param obj a object added
     * @return this writer
     */
    public final XmcWriter add(XcAppendable obj) throws XmException
    {
        if(obj != null)
            obj.appendCode(this);
        return this;
    }

    /**
     * Flushes output.
     */
    public final void flush()
    {
        _writer.flush();
    }

    /**
     * Closes output.
     */
    public final void close()
    {
        _writer.close();
    }

    @Override
    public final String toString()
    {
        if(_swriter != null)
            return _swriter.toString();
        return super.toString();
    }
}
