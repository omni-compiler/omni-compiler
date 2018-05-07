/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.Writer;

/**
 * Writer for Fortran source code.
 */
public class XmfWriter
{
    public static final String DEFAULT_SPC_CHARS = " ";
    // indent is 1
    public static final String DEFAULT_INDENT_CHARS = " ";

    public static final int MAX_INDENT_LEVEL = 8;

    //public static final int DEFAULT_MAX_COLUMN_COUNT = 72;
    // In F77 The maximum count of colomn is 72.

    public static final int DEFAULT_MAX_COLUMN_COUNT = 132;
    // In F90 The maximum count of colomn is 132.

    // for contiguous line symbol('&')
    public static final int RESERVE_COLUMN_COUNT = 2;

    private PrintWriter _out;

    /* Indent string */
    private String _indentChars = DEFAULT_INDENT_CHARS;
    /** Indent string length. */
    private int _indentCharsLength = DEFAULT_INDENT_CHARS.length();
    private int _maxColumnCount = DEFAULT_MAX_COLUMN_COUNT;
    private boolean _wrapStringLiteral = true;

    /*
     * Current state
     */
    private int _lineNumber = 0;
    private int _columnNumber = 0;
    private int _indentLevel = 0;
    private boolean _needSeparator = false;

    public enum StatementMode { FORTRAN, XMP, OMP, ACC }

    private StatementMode _mode = StatementMode.FORTRAN;

    public XmfWriter(PrintWriter writer)
    {
        _out = writer;
    }

    public void setStatementMode(StatementMode m)
    {
	_mode = m;
    }

    public StatementMode getStatementMode()
    {
	return _mode;
    }

    /**
     * Get column count of character glyph.
     * @param c Character.
     * @return 1 or 2
     */
    private static int _getColumnCount(char c)
    {
        // Character code considers less than 0x100
        // to be a halfwidth character.
        if (c < 0x0100) {
            return 1;
        }
        return 2;
    }

    private static int _getColumnCount(char[] array)
    {
        int columnCount = 0;
        for (char c : array) {
            columnCount += _getColumnCount(c);
        }
        return columnCount;
    }

    private static int _getColumnCount(String s)
    {
        return _getColumnCount(s.toCharArray());
    }

    /**
     * Judge line dirty.
     * @return true/false.
     */
    private boolean _isLineDirty()
    {
        if (_columnNumber == 0) {
            return false;
        }
        return true;
    }

    /**
     * Check line full.
     * @return true/false.
     */
    private boolean _checkLineFull(int tokenColumnCount)
    {
        if ((_columnNumber + tokenColumnCount) <= (_maxColumnCount - RESERVE_COLUMN_COUNT)) {
            return false;
        }
        return true;
    }

    /**
     * Write indent.
     */
    private void _writeIndent()
    {
        // line is dirty?
        if (_isLineDirty() != false) {
            return;
        }

        for (int i = 0; i < getIndentLevel(); i++) {
            _out.print(_indentChars);
            _columnNumber += _indentCharsLength;
        }
    }

    /**
     * Setup line.
     * @param writtingColumnCount Number of writting column count.
     */
    private void _setupLine(int writtingColumnCount)
    {
        if (_isLineDirty() == false) {
            _writeIndent();
        }
        else if (_checkLineFull(writtingColumnCount) != false) {
            _out.print("&");
            setupNewLine();
            _writeIndent();
	    if (_mode == StatementMode.OMP){
	      _out.print("!$OMP");
	      _columnNumber += 5;
	    }
	    else if (_mode == StatementMode.ACC){
	      _out.print("!$ACC");
	      _columnNumber += 5;
	    }
        }
    }

    /**
     * Convert to quoted string.
     * @param s Any string.
     * @return Converted string.
     */
    private static String _toStringLiteral(String s)
    {
        // Buffer capacity = <source length> * 2 + <literal prefix and suffix>
        StringBuilder buff = new StringBuilder(s.length() * 2 + 2);
        buff.append('"');
        for (char c : s.toCharArray()) {
            if (c == '"') {
                buff.append("\"\"");
            } else {
                buff.append(c);
            }
        }
        buff.append('"');
        return buff.toString();
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
     * Get current indent level.
     * @return indent level.
     */
    public int getIndentLevel()
    {
        // for safety
        if (_indentLevel < 0) {
            return 0;
        }
        if (_indentLevel > MAX_INDENT_LEVEL) {
            return MAX_INDENT_LEVEL;
        }
        return _indentLevel;
    }

    /**
     * Increment indent level.
     */
    public void incrementIndentLevel()
    {
        ++_indentLevel;
    }

    /**
     * Decrement indent level.
     */
    public void decrementIndentLevel()
    {
        --_indentLevel;
    }

    /**
     * Get current line number.
     * @return Line number.
     */
    public int getLineNumber()
    {
        return _lineNumber;
    }

    /**
     * Get writer set by this writer.
     * @return Writer.
     */
    public Writer getWriter()
    {
        return _out;
    }

    /**
     * Setup new line.
     */
    public void setupNewLine()
    {
        _out.println();
        _columnNumber = 0;
        _lineNumber++;
        _needSeparator = false;
        flush();
    }

    /**
     * Setup contiguous new line.
     */
    public void setupContiguousNewLine()
    {
        _out.print(" &");
        setupNewLine();
    }

    private boolean _isWhitespace(String s)
    {
        for (char c : s.toCharArray()) {
            if (!Character.isWhitespace(c))
                return false;
        }

        return true;
    }

    /**
     * Write in the designated string as token.
     * @param s String to write
     */
    public void writeToken(String s)
    {
        if (s == null || s.length() == 0 || _isWhitespace(s)) {
            return;
        }

        StringBuilder sb = new StringBuilder();

        if (_needSeparator)
            sb.append(' ');
        sb.append(s.trim());

        int columnCount = _getColumnCount(sb.toString());
        _setupLine(columnCount);

        _writeCharacterArray(sb.toString().toCharArray());
    }

    /**
     * Write in the designated string as literal string of Fortran.
     * @param s String to write
     */
    public void writeLiteralString(String s)
    {
        if (s == null || s.length() == 0) {
            s = "\"\"";
        } else {
            s = _toStringLiteral(s);
        }

        if (_needSeparator) {
            StringBuilder buf = new StringBuilder();
            buf.append(" ");
            buf.append(s);
            s = buf.toString();
        }

        char[] chArray = s.toCharArray();

        if (_wrapStringLiteral == false) {
            // prepare for area for 2 characters
            _setupLine(2);
        } else {
            // prepare for area for string
            _setupLine(_getColumnCount(chArray));
        }

        _writeCharacterArray(chArray);
    }

    private void _writeCharacterArray(char[] chArray)
    {
        int offset = 0;
        while (offset < chArray.length) {
            if (offset > 0) {
                _out.print("&");
                setupNewLine();
                _setupLine(1);
                _out.print("&");
                _columnNumber++;
            }

            int restColumnCount = _maxColumnCount - _columnNumber - RESERVE_COLUMN_COUNT;
            // for safety
            if (restColumnCount < 1) {
                restColumnCount = 1;
            }

            // System.out.print(restColumnCount);
            int substringLength = 0;
            int substringColumnCount = 0;

            for (int i = offset; i < chArray.length; i++) {
                int columnCount = _getColumnCount(chArray[i]);
                if (restColumnCount < (substringColumnCount + columnCount)) {
                    break;
                }
                substringColumnCount += columnCount;
                substringLength++;
            }

            // for safety
            if (substringLength < 1) {
                substringLength = 1;
            }

            _out.write(chArray, offset, substringLength);
            _columnNumber += substringColumnCount;
            offset += substringLength;
        }

        _needSeparator = true;
    }


    /**
     * Write in the designated string as block comment of Fortran.
     * @param s String to write
     */
    public void writeBlockComment(String s)
    {
        if (_isLineDirty() != false) {
            setupNewLine();
        }

        BufferedReader reader = new BufferedReader(new StringReader(s));
        String line;
        try {
            while ((line = reader.readLine()) != null) {
                // Note: Omit count up of _columnNumber
                _out.print("!");
                _out.print(line);
                setupNewLine();
            }
            reader.close();
        } catch (IOException e) {
            // Impossible.
            e.printStackTrace();
        }
    }

    /**
     * Write in the designated string as isolated line.
     * @param s String to write
     */
    public void writeIsolatedLine(String s)
    {
        if (_isLineDirty() != false) {
            setupNewLine();
        }

        // Note: Omit count up of _columnNumber
        _out.print(s);
        setupNewLine();
    }

    public void skipSeparator()
    {
      _needSeparator = false;
    }

    public void needSeparator()
    {
      _needSeparator = true;
    }

    /**
     * Flush writer.
     */
    public void flush()
    {
        _out.flush();
    }

    /**
     * Close writer.
     */
    public void close()
    {
        _out.close();
    }
}
