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

public class XmObjInterfaceMatcherTest
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
        XmObjInterfaceMatcher m = new XmObjInterfaceMatcher((String)null);
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_e2()
    {
        XmObjInterfaceMatcher m = new XmObjInterfaceMatcher("");
        assertFalse(m.match(forStmt));
    }

    @Test
    public void testMatch_e3()
    {
        XmObjInterfaceMatcher m = new XmObjInterfaceMatcher("Hoge");
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_s1()
    {
        XmObjInterfaceMatcher m = 
            new XmObjInterfaceMatcher("xcodeml.util.XmNode");
        assertTrue(m.match(forStmt));
    }
    
    @Test
    public void testMatch_s2()
    {
        XmObjInterfaceMatcher m = 
            new XmObjInterfaceMatcher("xcodeml.c.binding.gen.IXbcStatementsChoice");
        assertTrue(m.match(forStmt));
    }

    @Test
    public void testMatch_s3()
    {
        XmObjInterfaceMatcher m = 
            new XmObjInterfaceMatcher("java.lang.Cloneable");
        assertTrue(m.match(forStmt));
    }
    
    @Test
    public void testMatch_f1()
    {
        XmObjInterfaceMatcher m = 
            new XmObjInterfaceMatcher("xcodeml.util.XmNode");
        assertFalse(m.match(null));
    }

    @Test
    public void testMatch_f2()
    {
        XmObjInterfaceMatcher m = 
            new XmObjInterfaceMatcher("java.lang.Runnable");
        assertFalse(m.match(forStmt));
    }
    
    @Test
    public void testMatch_f3()
    {
        XmObjInterfaceMatcher m = 
            new XmObjInterfaceMatcher("xcodeml.c.obj.XmcForStatement");
        assertFalse(m.match(forStmt));
    }

}
