/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.util.XmNodeImpl;
import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

public class XmNodePostOrderIteratorTest
{
    XmNodeImpl tree_one;
    XmNodeImpl tree_five;
    

    @Before
    public void setUp() throws Exception
    {
        tree_one = new XmNodeImpl("a");
        tree_five = new XmNodeImpl("a");
        XmNodeImpl b = new XmNodeImpl("b");
        b.addChild(new XmNodeImpl("c"));
        b.addChild(new XmNodeImpl("d"));
        tree_five.addChild(b);
        tree_five.addChild(new XmNodeImpl("e"));
    }

    @Test
    public void testXmNodePostOrderIterator_null()
    {
        @SuppressWarnings("unused")
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(null);
    }
    
    @Test
    public void testXmNodePostOrderIterator_one()
    {
        @SuppressWarnings("unused")
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(tree_one);
    }    

    @Test
    public void testHasNext_null()
    {
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(null);
        assertFalse(it.hasNext());
    }

    @Test
    public void testHasNext_one()
    {
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(tree_one);
        assertTrue(it.hasNext());
        it.next();
        assertFalse(it.hasNext());
    }
    
    @Test
    public void testHasNext_five()
    {
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(tree_five);
        for (int i = 0; i < 4; ++i) {
            assertTrue(it.hasNext());
            it.next();
        }
        it.next();
        assertFalse(it.hasNext());
    }
    
    @Test
    public void testNext_null()
    {
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(null);
        assertNull(it.next());
    }

    @Test
    public void testNext_one()
    {
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(tree_one);
        XmNodeImpl result = (XmNodeImpl)it.next();
        assertEquals("a", result.getName());
    }

    @Test
    public void testNext_five()
    {
        XmNodePostOrderIterator it = new XmNodePostOrderIterator(tree_five);
        XmNodeImpl result = (XmNodeImpl)it.next();
        assertEquals("c", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("d", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("b", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("e", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("a", result.getName());        
    }
    
    @Test
    public void testRemove()
    {
        try {
            XmNodePostOrderIterator it = new XmNodePostOrderIterator(tree_one);
            it.remove();
            fail();
        } catch (UnsupportedOperationException e) {
        }
    }

}
