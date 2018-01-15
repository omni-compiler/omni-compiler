package xcodeml.c.type;

import java.util.ArrayList;
import java.util.Iterator;
import xcodeml.util.XmException;

/**
 * list of XmType
 */
public class XcTypeList implements Iterable<XcType>
{
    private ArrayList<XcType> _list = new ArrayList<XcType>();
    
    public XcTypeList()
    {
    }

    public boolean add(XcType e)
    {
        return _list.add(e);
    }

    public final void clear()
    {
        _list.clear();
    }

    public final boolean contains(XcType type)
    {
        return _list.contains(type);
    }

    public final XcType get(int index)
    {
        return _list.get(index);
    }

    public final boolean isEmpty()
    {
        return _list.isEmpty();
    }

    @Override
    public final Iterator<XcType> iterator()
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
    
    public final void resolve(XcIdentTableStack itStack) throws XmException
    {
        for(XcType type : _list)
            type.resolve(itStack);
    }
    
    @Override
    public String toString()
    {
        return _list.toString();
    }
}
