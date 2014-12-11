/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import java.util.Iterator;

import xcodeml.XmException;
import xcodeml.c.util.XmcWriter;

/**
 * struct/union member list (element type is XmIdent)
 */
public final class XcMemberList extends XcIdentList
{
    private XcType _parentType;
    
    public XcMemberList(XcType parentType)
    {
        _parentType = parentType;
    }

    @Override
    public final void add(XcIdent e)
    {
        super.add(e);
        e.setVarKindEnum(XcVarKindEnum.MEMBER);
        e.setParentType(_parentType);
    }

    public final void appendCode(XmcWriter w) throws XmException
    {
        Iterator<XcIdent> ite = iterator();

        w.spc().add('{').lf();

        for(; ite.hasNext();) {
            XcIdent ident = ite.next();
            String symbol = ident.getSymbol();
            XcType type = ident.getType();

            ident.appendGccExtension(w);

            type.appendDeclCode(w, symbol, true, false);

            if(ident.getIsBitField()) {
                w.add(":");
                if(ident.getIsBitFieldExpr()) {
                    w.add(ident.getBitFieldExpr());
                } else {
                    w.add(ident.getBitField());
                }
            }

            w.add(ident.getGccAttribute());

            w.add(';').lf();
        }
        
        w.add('}');
    }
}
