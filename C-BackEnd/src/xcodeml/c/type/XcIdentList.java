/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import java.util.ArrayList;
import java.util.Iterator;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;

/**
 * List of XcIdent.
 */
public class XcIdentList implements Cloneable, Iterable<XcIdent>, XcNode
{
    private ArrayList<XcIdent> _list = new ArrayList<XcIdent>();
    
    public XcIdentList()
    {
    }
    
    public XcIdentList copy()
    {
        try {
            ArrayList<XcIdent> list = new ArrayList<XcIdent>();
            list.addAll(_list);
            XcIdentList obj = (XcIdentList)clone();
            obj._list = list;
            return obj;
        } catch(CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public void add(XcIdent e)
    {
        _list.add(e);
    }

    public final boolean contains(XcIdent e)
    {
        return _list.contains(e);
    }

    public final XcIdent get(int index)
    {
        return _list.get(index);
    }
    
    public final void set(int index, XcIdent ident)
    {
        _list.set(index, ident);
    }
    
    public final boolean isEmpty()
    {
        return _list.isEmpty();
    }
    
    public final void clear()
    {
        _list.clear();
    }

    @Override
    public final Iterator<XcIdent> iterator()
    {
        return _list.iterator();
    }

    public final int size()
    {
        return _list.size();
    }

    public final <T> T[] toArray(T[] a)
    {
        return _list.toArray(a);
    }

    public final XcIdent getBySymbol(String symbol) 
    {
        for(Iterator<XcIdent> ite = _list.iterator(); ite.hasNext();) {
            XcIdent ident = ite.next();
            if(ident.getSymbol().equals(symbol))
                return ident;
        }
        return null;
    }

    public final boolean isVoid()
    {
        if(isEmpty() || size() >= 2)
            return false;
        
        XcType type = get(0).getType();
        if(type == null)
            throw new IllegalStateException();
        if(XcTypeEnum.BASETYPE.equals(type.getTypeEnum()) == false)
            return false;
        
        return XcBaseTypeEnum.VOID.equals(((XcBaseType)type).getBaseTypeEnum());
    }

    public final void resolve(XcIdentTableStack itStack) throws XmException
    {
        for(XcIdent ident : _list)
            ident.resolve(itStack);
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcIdent)
            _list.add((XcIdent)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        if(_list.isEmpty())
            return null;
        return _list.toArray(new XcNode[_list.size()]);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        if(index < _list.size())
            _list.set(index, (XcIdent)child);
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public String toString()
    {
        return _list.toString();
    }
}
