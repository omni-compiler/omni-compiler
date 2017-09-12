package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   pragma, text, statement
 */
public class XcDirectiveObj extends XcStmtObj implements XcDecAndDefObj
{
    private String _line;
    private XcStmtObj _stmt = null;

    /**
     * Creates XcDirectiveObj.
     */
    public XcDirectiveObj()
    {
    }

    /**
     * Creates XcDirectiveObj.
     * 
     * @param line a content of the directive.
     */
    public XcDirectiveObj(String line)
    {
        _line = line;
    }

    /**
     * Sets a content of the directive.
     * 
     * @param line a content of the directive.
     */
    public final void setLine(String line)
    {
        _line = line;
    }

    public final void addToken(String line)
    {
	_line += " " + line;
    }

    @Override
    public void addChild(XcNode child)
    {
       if(child instanceof XcExprObj){
           XmcWriter w = new XmcWriter();
           try{
               ((XcExprObj)child).appendCode(w);
               _line += " " + w.toString();
           }catch(XmException xme){
               throw new RuntimeException(xme.getMessage());
           }
       }else if(child instanceof XcStmtObj){
           if(_stmt == null){
               _stmt = (XcStmtObj)child;
           }else {
               throw new IllegalArgumentException(child.getClass().getName());
           }
	}else {
	    throw new IllegalArgumentException(child.getClass().getName());
	}
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return null;
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }
    
    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        super.appendCode(w);
        w.noLfOrLf();
        w.add(_line).lf();
        w.add(_stmt);
    }
}
