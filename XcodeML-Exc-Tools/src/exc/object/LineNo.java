package exc.object;

import xcodeml.util.ILineNo;

public class LineNo implements ILineNo
{
    String fname;
    int ln;
  String line_prefix;

    static LineNo current_lineno;

    public LineNo(String fname, int ln)
    {
        this.fname = fname.intern();
        this.ln = ln;
        this.line_prefix = null;
    }

    public int lineNo()
    {
        return ln;
    }

    public String fileName()
    {
        return fname;
    }

  public void setLinePrefix(String prefix)
  {
    line_prefix = prefix;
  }

  public String getLinePrefix()
  {
    return line_prefix;
  }

    @Override
    public String toString()
    {
        return (fname != null ? fname : "") + ":" + ln;
    }
    
    public static LineNo make(String fname, int ln)
    {
        if(current_lineno == null || current_lineno.ln != ln || !current_lineno.fname.equals(fname)) {
            if(fname == null) {
                if(current_lineno == null)
                    fname = "*";
                else
                    fname = current_lineno.fname;
            }
            current_lineno = new LineNo(fname, ln);
        }
        return current_lineno;
    }
}
