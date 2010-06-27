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
 * enumerator list (element type is XmIdent)
 */
public final class XcEnumeratorList extends XcIdentList
{
    private XcType _parentType;
    
    public XcEnumeratorList(XcType parentType)
    {
        _parentType = parentType;
    }

    @Override
    public final void add(XcIdent e)
    {
        super.add(e);
        e.setSymbolKindEnum(XcSymbolKindEnum.MOE);
        e.setParentType(_parentType);
    }

    public final void appendCode(XmcWriter w) throws XmException
    {
        Iterator<XcIdent> ite = iterator();

        if(ite.hasNext() == false)
            return;

        w.spc().add('{').lf();

        while(ite.hasNext()) {
            XcIdent ident = ite.next();
            ident.appendInitCode(w, false);


            if(ite.hasNext())
                w.add(',');
            w.lf();
        }

        w.add('}');
    }
}
