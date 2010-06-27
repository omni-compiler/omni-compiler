/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Iterator for XobjList
 */
public class XobjListIterator implements Iterator<Xobject>
{
    private XobjArgs _current;
    
    XobjListIterator(XobjList xobjList)
    {
        _current = xobjList.getArgs();
    }
    
    @Override
    public boolean hasNext()
    {
        return _current != null;
    }

    @Override
    public Xobject next()
    {
        if(_current == null)
            throw new NoSuchElementException();
        
        Xobject obj = _current.getArg();
        _current = _current.next;
        
        return obj;
    }

    @Override
    public void remove()
    {
        throw new UnsupportedOperationException();
    }
}
