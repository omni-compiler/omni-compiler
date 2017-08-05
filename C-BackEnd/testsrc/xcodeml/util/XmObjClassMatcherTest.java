/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import xcodeml.c.obj.XmcForStatement;

public class XmObjClassMatcherTest
{
    XmcForStatement forStmt;

    @Before
    public void setUp() throws Exception
    {
        forStmt = new XmcForStatement();
    }

    @Test
    public void testMatch_e1()
    {
        XmObjClassMatcher m = new XmObjClassMatcher((String)null);
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_e2()
    {
        XmObjClassMatcher m = new XmObjClassMatcher("");
        assertFalse(m.match(forStmt));
    }

    @Test
    public void testMatch_e3()
    {
        XmObjClassMatcher m = new XmObjClassMatcher("Hoge");
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_s1()
    {
        XmObjClassMatcher m = 
            new XmObjClassMatcher("xcodeml.c.obj.XmcForStatement");
        assertTrue(m.match(forStmt));
    }

    @Test
    public void testMatch_f1()
    {
        XmObjClassMatcher m = 
            new XmObjClassMatcher("xcodeml.c.obj.XmcForStatement");
        assertFalse(m.match(null));
    }

    @Test
    public void testMatch_f2()
    {
        XmObjClassMatcher m = 
            new XmObjClassMatcher("java.lang.Object");
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_f3()
    {
        XmObjClassMatcher m = 
            new XmObjClassMatcher("xcodeml.c.obj.XmcWhileStatement");
        assertFalse(m.match(forStmt));
    }

}
