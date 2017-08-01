/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import static org.junit.Assert.*;

import java.io.StringReader;
import java.util.List;

import org.junit.Test;

import xcodeml.c.binding.gen.XbcXcodeProgram;
import xcodeml.c.decompile.XcBindingVisitor;
import xcodeml.c.decompile.XcDecAndDefObj;
import xcodeml.c.decompile.XcDeclObj;
import xcodeml.c.decompile.XcProgramObj;
import xcodeml.c.util.XmcWriter;

public class XbcBindingVisitorTest
{
    private XbcXcodeProgram _createProgram(String xml) throws Exception
    {
        XbcXcodeProgram xprog =new XbcXcodeProgram(new StringReader(xml));
        return xprog;
    }
    
    @Test
    public void test_XbStartXcodeProgram() throws Exception
    {
        String xml =
            "<XcodeProgram>" +
            "  <typeTable>" +
            "  </typeTable>" +
            "  <globalSymbols>" +
            "    <id><name type=\"long_long\">a</name></id>" +
            "    <id><name type=\"double\">b</name></id>" +
            "  </globalSymbols>" +
            "  <globalDeclarations>" +
            "    <varDecl>" +
            "      <name>a</name>" +
            "      <value><intConstant>1</intConstant></value>" +
            "    </varDecl>" +
            "    <varDecl>" +
            "      <name>b</name>" +
            "      <value><floatConstant>0x3FF00000 0x0</floatConstant></value>" +
            "    </varDecl>" +
            "  </globalDeclarations>" +
            "</XcodeProgram>" +
            "";

        XbcXcodeProgram xprog = _createProgram(xml);
        
        XcProgramObj prog = XcBindingVisitor.createXcProgramObj(xprog);
        assertNotNull(prog);
        List<XcDecAndDefObj> declAndDefs = prog.getDeclAndDefList();
        assertEquals(2, declAndDefs.size());
        XcDecAndDefObj declAndDef = declAndDefs.get(0);
        assertTrue(declAndDef instanceof XcDeclObj);

        XmcWriter w = new XmcWriter();
        prog.appendCode(w);
        System.out.println(w.toString());
    }
}
