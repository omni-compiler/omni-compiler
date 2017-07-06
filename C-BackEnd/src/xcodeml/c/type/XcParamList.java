package xcodeml.c.type;

import java.util.Iterator;
import xcodeml.util.XmException;
import xcodeml.c.util.XmcWriter;

/**
 * function parameter list (element type is XmIdent)
 */
public final class XcParamList extends XcIdentList
{
    private boolean _isEllipsised;

    public XcParamList()
    {
        _isEllipsised = false;
    }

    public boolean isEllipsised()
    {
        return _isEllipsised;
    }
    
    public void setIsEllipsised(boolean enable)
    {
        _isEllipsised = enable;
    }
    
    @Override
    public final void add(XcIdent ident)
    {
        ident.setSymbolKindEnum(XcSymbolKindEnum.VAR);
        ident.setVarKindEnum(XcVarKindEnum.PARAM);
        super.add(ident);
    }

    public final void appendArgs(XmcWriter w, boolean paramSymbol, boolean isPreDecl) throws XmException
    {
        w.add("(");

        if(isEmpty() == false) {
            for(Iterator<XcIdent> ite = iterator(); ite.hasNext();) {
                XcIdent ident = ite.next();
                String symbol = (paramSymbol ? ident.getSymbol() : null);
                ident.getType().appendDeclCode(w, symbol, true, isPreDecl);
                if(ite.hasNext())
                    w.add(",");
            }
        }

        if(_isEllipsised)
            w.add(", ...");
        
        w.add(")");
    }
}
