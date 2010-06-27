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

public class XmNodePreOrderIteratorTest
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
    public void testXmNodePreOrderIterator_null()
    {
        @SuppressWarnings("unused")
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(null);
    }
    
    @Test
    public void testXmNodePreOrderIterator_one()
    {
        @SuppressWarnings("unused")
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(tree_one);
    }    

    @Test
    public void testHasNext_null()
    {
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(null);
        assertFalse(it.hasNext());
    }

    @Test
    public void testHasNext_one()
    {
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(tree_one);
        assertTrue(it.hasNext());
        it.next();
        assertFalse(it.hasNext());
    }
    
    @Test
    public void testHasNext_five()
    {
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(tree_five);
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
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(null);
        assertNull(it.next());
    }

    @Test
    public void testNext_one()
    {
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(tree_one);
        XmNodeImpl result = (XmNodeImpl)it.next();
        assertEquals("a", result.getName());
    }

    @Test
    public void testNext_five()
    {
        XmNodePreOrderIterator it = new XmNodePreOrderIterator(tree_five);
        XmNodeImpl result = (XmNodeImpl)it.next();
        assertEquals("a", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("b", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("c", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("d", result.getName());
        result = (XmNodeImpl)it.next();
        assertEquals("e", result.getName());        
    }
    
    @Test
    public void testRemove()
    {
        try {
            XmNodePreOrderIterator it = new XmNodePreOrderIterator(tree_one);
            it.remove();
            fail();
        } catch (UnsupportedOperationException e) {
        }
    }

}
