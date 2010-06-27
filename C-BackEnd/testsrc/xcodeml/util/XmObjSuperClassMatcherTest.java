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

public class XmObjSuperClassMatcherTest
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
        XmObjSuperClassMatcher m = new XmObjSuperClassMatcher((String)null);
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_e2()
    {
        XmObjSuperClassMatcher m = new XmObjSuperClassMatcher("");
        assertFalse(m.match(forStmt));
    }

    @Test
    public void testMatch_e3()
    {
        XmObjSuperClassMatcher m = new XmObjSuperClassMatcher("Hoge");
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_s1()
    {
        XmObjSuperClassMatcher m = 
            new XmObjSuperClassMatcher("xcodeml.c.obj.XmcForStatement");
        assertTrue(m.match(forStmt));
    }
    
    @Test
    public void testMatch_s2()
    {
        XmObjSuperClassMatcher m = 
            new XmObjSuperClassMatcher("xcodeml.c.binding.XbcObj");
        assertTrue(m.match(forStmt));
    }
    
    @Test
    public void testMatch_s3()
    {
        XmObjSuperClassMatcher m = 
            new XmObjSuperClassMatcher("java.lang.Object");
        assertTrue(m.match(forStmt));
    }

    @Test
    public void testMatch_f1()
    {
        XmObjSuperClassMatcher m = 
            new XmObjSuperClassMatcher("xcodeml.c.obj.XmcForStatement");
        assertFalse(m.match(null));
    }
    
    @Test
    public void testMatch_f3()
    {
        XmObjSuperClassMatcher m = 
            new XmObjSuperClassMatcher("xcodeml.c.obj.XmcWhileStatement");
        assertFalse(m.match(forStmt));
    }

}
