/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.Before;

import java.util.*;
import java.io.*;

import xcodeml.c.binding.gen.XbcXcodeProgram;
import xcodeml.c.decompile.XcBindingVisitor;
import xcodeml.c.decompile.XcProgramObj;
import xcodeml.c.util.XmcValidator;
import xcodeml.c.util.XmcWriter;

public class XmExpressionsTest {

    public static final String DATAPATH = "testdata";

    private XmValidator validator;
    private XmcWriter writer;
    private XbcXcodeProgram xprog;
    private List<String> errorList;

    public boolean compare(String output, BufferedReader expected) throws Exception
    {
        output = output.trim();

        int i;
        String expectedLine;
        String outputLine;
        String[] outputLines = output.split("\n");

        for(i = 0; i < outputLines.length; i++){
            outputLine = outputLines[i].trim();
            if((expectedLine = expected.readLine()) == null) {
                System.out.println("fail at line" + i);
                System.out.println("output   : " + outputLine);
                return false;
            }

            expectedLine = expectedLine.trim();

            if (!(expectedLine.equals(outputLine))) {
                System.out.println("fail at line" + i);
                System.out.println("expected : " + expectedLine);
                System.out.println("output   : " + outputLine);
                return false;
            }
        }

        if ((expectedLine = expected.readLine()) != null) {
            expectedLine = expectedLine.trim();
            if (expectedLine.length() != 0 &&
                expected.readLine() != null) {
                System.out.println("fail at line" + i);
                System.out.println("expected : " + expectedLine);
                return false;
            }
        }

        return true;
    }


	@Before
	public void setUp() throws Exception
    {
        validator = new XmcValidator();
        writer = new XmcWriter();
        xprog = new XbcXcodeProgram();
        errorList = new ArrayList<String>();
	}

    public void runTest(String target) throws Exception
    {
        XcProgramObj prog;
        
        errorList.clear();
        
        String targetPath = DATAPATH + File.separator + target;
        String inputPath = targetPath + ".xml";
        String expectedPath = targetPath + ".c";
        
        FileReader inputFile = null;
        FileReader expectedFile = null;

        try {
            inputFile = new FileReader(inputPath);
            expectedFile = new FileReader(expectedPath);

            BufferedReader input = new BufferedReader(inputFile);
            BufferedReader expected = new BufferedReader(expectedFile);

            assertTrue(validator.read(input, xprog, errorList));

            prog = XcBindingVisitor.createXcProgramObj(xprog);
            prog.appendCode(writer);

            assertTrue(compare(writer.toString(), expected));
        } catch (Exception e) {
        } finally {
            if (inputFile != null) {
                try {
                    inputFile.close();
                } catch (Exception e) {
                }
            }
            if (expectedFile != null) {
                try {
                    expectedFile.close();
                } catch (Exception e) {
                }
            }
        }
    }

    @Test
    public void test_LshiftExpr() throws Exception
    {
        String target = "LshiftExpr";

        runTest(target);
    }

    @Test
    public void test_RshiftExpr() throws Exception
    {
        String target = "RshiftExpr";

        runTest(target);
    }

    @Test
    public void test_array() throws Exception
    {
        String target = "array";

        runTest(target);
    }

    @Test
    public void test_asgBitAndExpr() throws Exception
    {
        String target = "asgBitAndExpr";

        runTest(target);
    }

    @Test
    public void test_asgBitOrExpr() throws Exception
    {
        String target = "asgBitOrExpr";

        runTest(target);
    }

    @Test
    public void test_asgBitXorExpr() throws Exception
    {
        String target = "asgBitXorExpr";

        runTest(target);
    }

    @Test
    public void test_asgDivExpr() throws Exception
    {
        String target = "asgDivExpr";

        runTest(target);
    }

    @Test
    public void test_asgLshiftExpr() throws Exception
    {
        String target = "asgLshiftExpr";

        runTest(target);
    }

    @Test
    public void test_asgMinusExpr() throws Exception
    {
        String target = "asgMinusExpr";

        runTest(target);
    }

    @Test
    public void test_asgModExpr() throws Exception
    {
        String target = "asgModExpr";

        runTest(target);
    }

    @Test
    public void test_asgMulExpr() throws Exception
    {
        String target = "asgMulExpr";

        runTest(target);
    }

    @Test
    public void test_asgPlusExpr() throws Exception
    {
        String target = "asgPlusExpr";

        runTest(target);
    }

    @Test
    public void test_asgRshiftExpr() throws Exception
    {
        String target = "asgRshiftExpr";

        runTest(target);
    }

    @Test
    public void test_assignExpr() throws Exception
    {
        String target = "assignExpr";

        runTest(target);
    }

    @Test
    public void test_bitAndExpr() throws Exception
    {
        String target = "bitAndExpr";

        runTest(target);
    }

    @Test
    public void test_bitOrExpr() throws Exception
    {
        String target = "bitOrExpr";

        runTest(target);
    }

    @Test
    public void test_bitXorExpr() throws Exception
    {
        String target = "bitXorExpr";

        runTest(target);
    }

    @Test
    public void test_breakStatement() throws Exception
    {
        String target = "breakStatement";

        runTest(target);
    }

    @Test
    public void test_castExpr() throws Exception
    {
        String target = "castExpr";

        runTest(target);
    }

    @Test
    public void test_commaExpr() throws Exception
    {
        String target = "commaExpr";

        runTest(target);
    }

    @Test
    public void test_commaExpr2() throws Exception
    {
        String target = "commaExpr2";

        runTest(target);
    }

    @Test
    public void test_contiuneStatement() throws Exception
    {
        String target = "contiuneStatement";

        runTest(target);
    }

    @Test
    public void test_divExpr() throws Exception
    {
        String target = "divExpr";

        runTest(target);
    }

    @Test
    public void test_doWhileStatement() throws Exception
    {
        String target = "doWhileStatement";

        runTest(target);
    }

    @Test
    public void test_enumType() throws Exception
    {
        String target = "enumType";

        runTest(target);
    }

    @Test
    public void test_externDecl() throws Exception
    {
        String target = "externDecl";

        runTest(target);
    }

    @Test
    public void test_floatConstant() throws Exception
    {
        String target = "floatConstant";

        runTest(target);
    }

    @Test
    public void test_forStatement() throws Exception
    {
        String target = "forStatement";

        runTest(target);
    }

    @Test
    public void test_functionDefinition() throws Exception
    {
        String target = "functionDefinition";

        runTest(target);
    }

    @Test
    public void test_functionPointer() throws Exception
    {
        String target = "functionPointer";

        runTest(target);
    }

    @Test
    public void test_gotoStatement() throws Exception
    {
        String target = "gotoStatement";

        runTest(target);
    }

    @Test
    public void test_ifStatement() throws Exception
    {
        String target = "ifStatement";

        runTest(target);
    }

    @Test
    public void test_ifelseStatement() throws Exception
    {
        String target = "ifelseStatement";

        runTest(target);
    }

    @Test
    public void test_intConstant() throws Exception
    {
        String target = "intConstant";

        runTest(target);
    }

    @Test
    public void test_logAndExpr() throws Exception
    {
        String target = "logAndExpr";

        runTest(target);
    }

    @Test
    public void test_logEQExpr() throws Exception
    {
        String target = "logEQExpr";

        runTest(target);
    }

    @Test
    public void test_logGEExpr() throws Exception
    {
        String target = "logGEExpr";

        runTest(target);
    }

    @Test
    public void test_logGTExpr() throws Exception
    {
        String target = "logGTExpr";

        runTest(target);
    }

    @Test
    public void test_logLEExpr() throws Exception
    {
        String target = "logLEExpr";

        runTest(target);
    }

    @Test
    public void test_logLTExpr() throws Exception
    {
        String target = "logLTExpr";

        runTest(target);
    }

    @Test
    public void test_logNEQExpr() throws Exception
    {
        String target = "logNEQExpr";

        runTest(target);
    }

    @Test
    public void test_logNotExpr() throws Exception
    {
        String target = "logNotExpr";

        runTest(target);
    }

    @Test
    public void test_logOrExpr() throws Exception
    {
        String target = "logOrExpr";

        runTest(target);
    }

    @Test
    public void test_memberArray() throws Exception
    {
        String target = "memberArray";

        runTest(target);
    }

    @Test
    public void test_minusExpr() throws Exception
    {
        String target = "minusExpr";

        runTest(target);
    }

    @Test
    public void test_modExpr() throws Exception
    {
        String target = "modExpr";

        runTest(target);
    }

    @Test
    public void test_mulExpr() throws Exception
    {
        String target = "mulExpr";

        runTest(target);
    }

    @Test
    public void test_nestedExpr() throws Exception
    {
        String target = "nestedExpr";

        runTest(target);
    }

    @Test
    public void test_operator() throws Exception
    {
        String target = "operator";

        runTest(target);
    }

    @Test
    public void test_plusExpr() throws Exception
    {
        String target = "plusExpr";

        runTest(target);
    }

    @Test
    public void test_postDecrExpr() throws Exception
    {
        String target = "postDecrExpr";

        runTest(target);
    }

    @Test
    public void test_postIncrExpr() throws Exception
    {
        String target = "postIncrExpr";

        runTest(target);
    }

    @Test
    public void test_primitiveTypes() throws Exception
    {
        String target = "primitiveTypes";

        runTest(target);
    }

    @Test
    public void test_struct() throws Exception
    {
        String target = "struct";

        runTest(target);
    }

    @Test
    public void test_switchStatement() throws Exception
    {
        String target = "switchStatement";

        runTest(target);
    }

    @Test
    public void test_unaryMinusExpr() throws Exception
    {
        String target = "unaryMinusExpr";

        runTest(target);
    }

    @Test
    public void test_union() throws Exception
    {
        String target = "union";

        runTest(target);
    }

    @Test
    public void test_preIncrExpr() throws Exception
    {
        String target = "preIncrExpr";

        runTest(target);
    }
    
    @Test
    public void test_preDecrExpr() throws Exception
    {
        String target = "preDecrExpr";

        runTest(target);
    }
    
    @Test
    public void test_condExpr() throws Exception
    {
        String target = "condExpr";

        runTest(target);
    }
    
    @Test
    public void test_sizeOfExpr() throws Exception
    {
        String target = "sizeOf";

        runTest(target);
    }
    
    @Test
    public void test_gccAlignOfExpr() throws Exception
    {
        String target = "gccAlignOf";

        runTest(target);
    }
}

