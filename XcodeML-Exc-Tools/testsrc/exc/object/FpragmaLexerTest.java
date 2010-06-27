/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import exc.xcodeml.XmfXmObjToXobjectTranslator;

import xcodeml.XmLanguage;
import xcodeml.f.binding.gen.XbfBody;
import xcodeml.f.binding.gen.XbfFfunctionDefinition;
import xcodeml.f.binding.gen.XbfFpragmaStatement;
import xcodeml.f.binding.gen.XbfId;
import xcodeml.f.binding.gen.XbfName;
import xcodeml.f.binding.gen.XbfSymbols;
import xcodeml.f.binding.gen.XbfTypeTable;
import xcodeml.util.XmOption;
import xcodeml.util.XmXmObjToXobjectTranslator;

public class FpragmaLexerTest
{
    private XmXmObjToXobjectTranslator _translator;
    
    @Before
    public void before() throws Exception
    {
        XmOption.setIsOpenMP(true);
        XmOption.setLanguage(XmLanguage.F);
        XobjectFile xobjFile = new XobjectFile();
        XbfTypeTable xtypeTable = new XbfTypeTable();
        _translator = new XmfXmObjToXobjectTranslator(xobjFile, xtypeTable);
    }
    
    @Test
    public void test_omp_parallel() throws Exception
    {
        XbfFpragmaStatement xpragma = new XbfFpragmaStatement();
        xpragma.setContent("omp parallel");
        FpragmaLexer lexer = new FpragmaLexer(_translator, xpragma);
        FpragmaLexer.Result result = lexer.lex(xpragma.getContent());
        assertNotNull(result);
        assertNull(result.error_message);
        Xobject x1 = result.xobject;
        assertNotNull(x1);
        assertEquals(Xcode.OMP_PRAGMA, x1.Opcode());
        Xobject x2 = x1.getArg(0);
        assertNotNull(x2);
        assertEquals(Xcode.STRING, x2.Opcode());
        Xobject x3 = x1.getArg(1);
        assertNotNull(x3);
        assertEquals(Xcode.STRING, x3.Opcode());
        assertEquals("PARALLEL", x3.getString());
    }

    @Test
    public void test_threadprivate() throws Exception
    {
        XbfFfunctionDefinition xdef = new XbfFfunctionDefinition();
        XbfSymbols xsymbols = new XbfSymbols();
        XbfId xid1 = new XbfId(), xid2 = new XbfId();
        XbfName xname1 = new XbfName(), xname2 = new XbfName();
        XbfBody xbody = new XbfBody();
        XbfFpragmaStatement xpragma = new XbfFpragmaStatement();
        
        xdef.setSymbols(xsymbols);
        xsymbols.addId(xid1);
        xsymbols.addId(xid2);
        xid1.setName(xname1);
        xid1.setSclass(XbfId.SCLASS_FSAVE);
        xname1.setContent("a");
        xid2.setName(xname2);
        xid2.setSclass(XbfId.SCLASS_FCOMMON_NAME);
        xname2.setContent("b");
        xdef.setBody(xbody);
        xbody.addDefModelStatement(xpragma);
        
        xpragma.setContent("omp threadprivate(a, /b/)");
        FpragmaLexer lexer = new FpragmaLexer(_translator, xpragma);
        FpragmaLexer.Result result = lexer.lex(xpragma.getContent());
        assertNotNull(result);
        assertEquals(null, result.error_message);
        Xobject x = result.xobject;
        assertNotNull(x);
        assertEquals(Xcode.OMP_PRAGMA, x.Opcode());
        Xobject x0 = x.getArg(0);
        assertNotNull(x0);
        assertEquals(Xcode.STRING, x0.Opcode());
        Xobject x1 = x.getArg(1);
        assertNotNull(x1);
        assertEquals(Xcode.STRING, x1.Opcode());
        assertEquals("THREADPRIVATE", x1.getString());
        Xobject x2 = x.getArg(2);
        assertNotNull(x2);
        assertEquals(Xcode.LIST, x2.Opcode());
        Xobject x20 = x2.getArg(0);
        assertNotNull(x20);
        assertEquals(Xcode.IDENT, x20.Opcode());
        assertEquals("a", x20.getName());
        Xobject x21 = x2.getArg(1);
        assertNotNull(x21);
        assertEquals(Xcode.IDENT, x21.Opcode());
        assertEquals("b", x21.getName());
    }

    @Test
    public void test_reduction_binop() throws Exception
    {
        XbfFfunctionDefinition xdef = new XbfFfunctionDefinition();
        XbfSymbols xsymbols = new XbfSymbols();
        XbfId xid1 = new XbfId(), xid2 = new XbfId(), xid3 = new XbfId();
        XbfName xname1 = new XbfName(), xname2 = new XbfName(), xname3 = new XbfName();
        XbfBody xbody = new XbfBody();
        XbfFpragmaStatement xpragma = new XbfFpragmaStatement();
        
        xdef.setSymbols(xsymbols);
        xsymbols.addId(xid1);
        xsymbols.addId(xid2);
        xsymbols.addId(xid3);
        xid1.setName(xname1);
        xid1.setSclass(XbfId.SCLASS_FLOCAL);
        xname1.setContent("a");
        xid2.setName(xname2);
        xid2.setSclass(XbfId.SCLASS_FLOCAL);
        xname2.setContent("b");
        xid3.setName(xname3);
        xid3.setSclass(XbfId.SCLASS_FLOCAL);
        xname3.setContent("c");
        xdef.setBody(xbody);
        xbody.addDefModelStatement(xpragma);
        
        xpragma.setContent("omp do reduction(+:a) reduction(*:b) reduction(-:c)");
        FpragmaLexer lexer = new FpragmaLexer(_translator, xpragma);
        FpragmaLexer.Result result = lexer.lex(xpragma.getContent());
        assertNotNull(result);
        assertEquals(null, result.error_message);
        Xobject x = result.xobject;
        assertNotNull(x);
        assertEquals(Xcode.OMP_PRAGMA, x.Opcode());
        Xobject x0 = x.getArg(0);
        assertNotNull(x0);
        assertEquals(Xcode.STRING, x0.Opcode());
        Xobject x1 = x.getArg(1);
        assertNotNull(x1);
        assertEquals(Xcode.STRING, x1.Opcode());
        assertEquals("FOR", x1.getString());
        Xobject x2 = x.getArg(2);
        assertNotNull(x2);
        assertEquals(Xcode.LIST, x2.Opcode());
        
        Xobject x20 = x2.getArg(0);
        assertEquals(Xcode.LIST, x20.Opcode());
        Xobject x200 = x20.getArg(0);
        assertEquals(Xcode.STRING, x200.Opcode());
        assertEquals("DATA_REDUCTION_PLUS", x200.getString());
        Xobject x201 = x20.getArg(1);
        assertEquals(Xcode.LIST, x201.Opcode());
        Xobject x2010 = x201.getArg(0);
        assertEquals(Xcode.IDENT, x2010.Opcode());
        assertEquals("a", x2010.getName());
        
        Xobject x21 = x2.getArg(1);
        assertEquals(Xcode.LIST, x21.Opcode());
        Xobject x210 = x21.getArg(0);
        assertEquals(Xcode.STRING, x210.Opcode());
        assertEquals("DATA_REDUCTION_MUL", x210.getString());
        Xobject x211 = x21.getArg(1);
        assertEquals(Xcode.LIST, x211.Opcode());
        Xobject x2110 = x211.getArg(0);
        assertEquals(Xcode.IDENT, x2110.Opcode());
        assertEquals("b", x2110.getName());

        Xobject x22 = x2.getArg(2);
        assertEquals(Xcode.LIST, x22.Opcode());
        Xobject x220 = x22.getArg(0);
        assertEquals(Xcode.STRING, x220.Opcode());
        assertEquals("DATA_REDUCTION_MINUS", x220.getString());
        Xobject x221 = x22.getArg(1);
        assertEquals(Xcode.LIST, x221.Opcode());
        Xobject x2210 = x221.getArg(0);
        assertEquals(Xcode.IDENT, x2210.Opcode());
        assertEquals("c", x2210.getName());
    }

    @Test
    public void test_reduction_logop() throws Exception
    {
        XbfFfunctionDefinition xdef = new XbfFfunctionDefinition();
        XbfSymbols xsymbols = new XbfSymbols();
        XbfId xid1 = new XbfId(), xid2 = new XbfId(), xid3 = new XbfId(), xid4 = new XbfId();
        XbfName xname1 = new XbfName(), xname2 = new XbfName(),
            xname3 = new XbfName(), xname4 = new XbfName();
        XbfBody xbody = new XbfBody();
        XbfFpragmaStatement xpragma = new XbfFpragmaStatement();
        
        xdef.setSymbols(xsymbols);
        xsymbols.addId(xid1);
        xsymbols.addId(xid2);
        xsymbols.addId(xid3);
        xsymbols.addId(xid4);
        xid1.setName(xname1);
        xid1.setSclass(XbfId.SCLASS_FLOCAL);
        xname1.setContent("a");
        xid2.setName(xname2);
        xid2.setSclass(XbfId.SCLASS_FLOCAL);
        xname2.setContent("b");
        xid3.setName(xname3);
        xid3.setSclass(XbfId.SCLASS_FLOCAL);
        xname3.setContent("c");
        xid4.setName(xname4);
        xid4.setSclass(XbfId.SCLASS_FLOCAL);
        xname4.setContent("d");
        xdef.setBody(xbody);
        xbody.addDefModelStatement(xpragma);
        
        xpragma.setContent("omp do reduction(.and.:a) reduction(.or.:b) reduction(.eqv.:c) reduction(.neqv.:d)");
        FpragmaLexer lexer = new FpragmaLexer(_translator, xpragma);
        FpragmaLexer.Result result = lexer.lex(xpragma.getContent());
        assertNotNull(result);
        assertEquals(null, result.error_message);
        Xobject x = result.xobject;
        assertNotNull(x);
        assertEquals(Xcode.OMP_PRAGMA, x.Opcode());
        Xobject x0 = x.getArg(0);
        assertNotNull(x0);
        assertEquals(Xcode.STRING, x0.Opcode());
        Xobject x1 = x.getArg(1);
        assertNotNull(x1);
        assertEquals(Xcode.STRING, x1.Opcode());
        assertEquals("FOR", x1.getString());
        Xobject x2 = x.getArg(2);
        assertNotNull(x2);
        assertEquals(Xcode.LIST, x2.Opcode());
        
        Xobject x20 = x2.getArg(0);
        assertEquals(Xcode.LIST, x20.Opcode());
        Xobject x200 = x20.getArg(0);
        assertEquals(Xcode.STRING, x200.Opcode());
        assertEquals("DATA_REDUCTION_LOGAND", x200.getString());
        Xobject x201 = x20.getArg(1);
        assertEquals(Xcode.LIST, x201.Opcode());
        Xobject x2010 = x201.getArg(0);
        assertEquals(Xcode.IDENT, x2010.Opcode());
        assertEquals("a", x2010.getName());

        Xobject x21 = x2.getArg(1);
        assertEquals(Xcode.LIST, x21.Opcode());
        Xobject x210 = x21.getArg(0);
        assertEquals(Xcode.STRING, x210.Opcode());
        assertEquals("DATA_REDUCTION_LOGOR", x210.getString());
        Xobject x211 = x21.getArg(1);
        assertEquals(Xcode.LIST, x211.Opcode());
        Xobject x2110 = x211.getArg(0);
        assertEquals(Xcode.IDENT, x2110.Opcode());
        assertEquals("b", x2110.getName());

        Xobject x22 = x2.getArg(2);
        assertEquals(Xcode.LIST, x22.Opcode());
        Xobject x220 = x22.getArg(0);
        assertEquals(Xcode.STRING, x220.Opcode());
        assertEquals("DATA_REDUCTION_LOGEQV", x220.getString());
        Xobject x221 = x22.getArg(1);
        assertEquals(Xcode.LIST, x221.Opcode());
        Xobject x2210 = x221.getArg(0);
        assertEquals(Xcode.IDENT, x2210.Opcode());
        assertEquals("c", x2210.getName());

        Xobject x23 = x2.getArg(3);
        assertEquals(Xcode.LIST, x23.Opcode());
        Xobject x230 = x23.getArg(0);
        assertEquals(Xcode.STRING, x230.Opcode());
        assertEquals("DATA_REDUCTION_LOGNEQV", x230.getString());
        Xobject x231 = x23.getArg(1);
        assertEquals(Xcode.LIST, x231.Opcode());
        Xobject x2310 = x231.getArg(0);
        assertEquals(Xcode.IDENT, x2310.Opcode());
        assertEquals("d", x2310.getName());
    }

    @Test
    public void test_reduction_intrinsic() throws Exception
    {
        XbfFfunctionDefinition xdef = new XbfFfunctionDefinition();
        XbfSymbols xsymbols = new XbfSymbols();
        XbfId xid1 = new XbfId(), xid2 = new XbfId(),
            xid3 = new XbfId(), xid4 = new XbfId(), xid5 = new XbfId();
        XbfName xname1 = new XbfName(), xname2 = new XbfName(),
            xname3 = new XbfName(), xname4 = new XbfName(), xname5 = new XbfName();
        XbfBody xbody = new XbfBody();
        XbfFpragmaStatement xpragma = new XbfFpragmaStatement();
        
        xdef.setSymbols(xsymbols);
        xsymbols.addId(xid1);
        xsymbols.addId(xid2);
        xsymbols.addId(xid3);
        xsymbols.addId(xid4);
        xsymbols.addId(xid5);
        xid1.setName(xname1);
        xid1.setSclass(XbfId.SCLASS_FLOCAL);
        xname1.setContent("a");
        xid2.setName(xname2);
        xid2.setSclass(XbfId.SCLASS_FLOCAL);
        xname2.setContent("b");
        xid3.setName(xname3);
        xid3.setSclass(XbfId.SCLASS_FLOCAL);
        xname3.setContent("c");
        xid4.setName(xname4);
        xid4.setSclass(XbfId.SCLASS_FLOCAL);
        xname4.setContent("d");
        xid5.setName(xname5);
        xid5.setSclass(XbfId.SCLASS_FLOCAL);
        xname5.setContent("e");
        xdef.setBody(xbody);
        xbody.addDefModelStatement(xpragma);
        
        xpragma.setContent("omp do reduction(min:a) reduction(max:b) reduction(iand:c) reduction(ior:d) reduction(ieor:e)");
        FpragmaLexer lexer = new FpragmaLexer(_translator, xpragma);
        FpragmaLexer.Result result = lexer.lex(xpragma.getContent());
        assertNotNull(result);
        assertEquals(null, result.error_message);
        Xobject x = result.xobject;
        assertNotNull(x);
        assertEquals(Xcode.OMP_PRAGMA, x.Opcode());
        Xobject x0 = x.getArg(0);
        assertNotNull(x0);
        assertEquals(Xcode.STRING, x0.Opcode());
        Xobject x1 = x.getArg(1);
        assertNotNull(x1);
        assertEquals(Xcode.STRING, x1.Opcode());
        assertEquals("FOR", x1.getString());
        Xobject x2 = x.getArg(2);
        assertNotNull(x2);
        assertEquals(Xcode.LIST, x2.Opcode());
        
        Xobject x20 = x2.getArg(0);
        assertEquals(Xcode.LIST, x20.Opcode());
        Xobject x200 = x20.getArg(0);
        assertEquals(Xcode.STRING, x200.Opcode());
        assertEquals("DATA_REDUCTION_MIN", x200.getString());
        Xobject x201 = x20.getArg(1);
        assertEquals(Xcode.LIST, x201.Opcode());
        Xobject x2010 = x201.getArg(0);
        assertEquals(Xcode.IDENT, x2010.Opcode());
        assertEquals("a", x2010.getName());

        Xobject x21 = x2.getArg(1);
        assertEquals(Xcode.LIST, x21.Opcode());
        Xobject x210 = x21.getArg(0);
        assertEquals(Xcode.STRING, x210.Opcode());
        assertEquals("DATA_REDUCTION_MAX", x210.getString());
        Xobject x211 = x21.getArg(1);
        assertEquals(Xcode.LIST, x211.Opcode());
        Xobject x2110 = x211.getArg(0);
        assertEquals(Xcode.IDENT, x2110.Opcode());
        assertEquals("b", x2110.getName());

        Xobject x22 = x2.getArg(2);
        assertEquals(Xcode.LIST, x22.Opcode());
        Xobject x220 = x22.getArg(0);
        assertEquals(Xcode.STRING, x220.Opcode());
        assertEquals("DATA_REDUCTION_IAND", x220.getString());
        Xobject x221 = x22.getArg(1);
        assertEquals(Xcode.LIST, x221.Opcode());
        Xobject x2210 = x221.getArg(0);
        assertEquals(Xcode.IDENT, x2210.Opcode());
        assertEquals("c", x2210.getName());

        Xobject x23 = x2.getArg(3);
        assertEquals(Xcode.LIST, x23.Opcode());
        Xobject x230 = x23.getArg(0);
        assertEquals(Xcode.STRING, x230.Opcode());
        assertEquals("DATA_REDUCTION_IOR", x230.getString());
        Xobject x231 = x23.getArg(1);
        assertEquals(Xcode.LIST, x231.Opcode());
        Xobject x2310 = x231.getArg(0);
        assertEquals(Xcode.IDENT, x2310.Opcode());
        assertEquals("d", x2310.getName());

        Xobject x24 = x2.getArg(4);
        assertEquals(Xcode.LIST, x24.Opcode());
        Xobject x240 = x24.getArg(0);
        assertEquals(Xcode.STRING, x240.Opcode());
        assertEquals("DATA_REDUCTION_IEOR", x240.getString());
        Xobject x241 = x24.getArg(1);
        assertEquals(Xcode.LIST, x241.Opcode());
        Xobject x2410 = x241.getArg(0);
        assertEquals(Xcode.IDENT, x2410.Opcode());
        assertEquals("e", x2410.getName());
    }

    @Test
    public void test_parallel_if() throws Exception
    {
        XbfFfunctionDefinition xdef = new XbfFfunctionDefinition();
        XbfSymbols xsymbols = new XbfSymbols();
        XbfId xid1 = new XbfId(), xid2 = new XbfId();
        XbfName xname1 = new XbfName(), xname2 = new XbfName();
        XbfBody xbody = new XbfBody();
        XbfFpragmaStatement xpragma = new XbfFpragmaStatement();
        
        xdef.setSymbols(xsymbols);
        xsymbols.addId(xid1);
        xsymbols.addId(xid2);
        xid1.setName(xname1);
        xid1.setSclass(XbfId.SCLASS_FSAVE);
        xid1.setType("Fint");
        xname1.setContent("a");
        xid2.setName(xname2);
        xid2.setSclass(XbfId.SCLASS_FSAVE);
        xid2.setType("Fint");
        xname2.setContent("b");
        xdef.setBody(xbody);
        xbody.addDefModelStatement(xpragma);
        
        xpragma.setContent("omp parallel if((a + max(b)) > 0 .and. .false.)");
        FpragmaLexer lexer = new FpragmaLexer(_translator, xpragma);
        FpragmaLexer.Result result = lexer.lex(xpragma.getContent());
        assertNotNull(result);
        assertEquals(null, result.error_message);
        Xobject x = result.xobject;
        assertNotNull(x);
        assertEquals(Xcode.OMP_PRAGMA, x.Opcode());
        Xobject x0 = x.getArg(0);
        assertNotNull(x0);
        assertEquals(Xcode.STRING, x0.Opcode());
        Xobject x1 = x.getArg(1);
        assertNotNull(x1);
        assertEquals(Xcode.STRING, x1.Opcode());
        assertEquals("PARALLEL", x1.getString());
        Xobject x2 = x.getArg(2);
        assertNotNull(x2);
        assertEquals(Xcode.LIST, x2.Opcode());
        Xobject x20 = x2.getArg(0);
        assertNotNull(x20);
        assertEquals(Xcode.LIST, x20.Opcode());
        Xobject x200 = x20.getArg(0);
        assertNotNull(x200);
        assertEquals(Xcode.STRING, x200.Opcode());
        assertEquals("DIR_IF", x200.getString());
        Xobject x201 = x20.getArg(1);
        assertNotNull(x201);
        assertEquals(Xcode.LOG_AND_EXPR, x201.Opcode());
    }

    @Test
    public void test_end_parallel() throws Exception
    {
        final String[][] pragmas = {
            { "omp end parallel",           "PARALLEL", null },
            { "omp end parallel do",        "PARALLEL_FOR", null },
            { "omp end parallel sections",  "PARALLEL_SECTIONS", null },
            { "omp end do",                 "FOR", null },
            { "omp end do nowait",          "FOR", "DIR_NOWAIT" },
            { "omp end sections",           "SECTIONS", null },
            { "omp end sections nowait",    "SECTIONS", "DIR_NOWAIT" },
            { "omp end single",             "SINGLE", null },
            { "omp end single nowait",      "SINGLE", "DIR_NOWAIT" },
            { "omp end critical",           "CRITICAL", null },
            { "omp end master",             "MASTER", null },
            { "omp end ordered",            "ORDERED", null },
        };

        for(int i = 0; i < pragmas.length; ++i) {
            
            String pragma = pragmas[i][0];
            String expectedClause = pragmas[i][1];
            String expectedArg = pragmas[i][2];
            
            XbfFfunctionDefinition xdef = new XbfFfunctionDefinition();
            XbfBody xbody = new XbfBody();
            XbfFpragmaStatement xpragma = new XbfFpragmaStatement();
            
            xdef.setBody(xbody);
            xbody.addDefModelStatement(xpragma);
            xpragma.setContent(pragma);
            
            FpragmaLexer lexer = new FpragmaLexer(_translator, xpragma);
            FpragmaLexer.Result result = lexer.lex(xpragma.getContent());
            assertNotNull(result);
            assertEquals(null, result.error_message);
            Xobject x = result.xobject;
            assertNotNull(x);
            assertEquals(Xcode.OMP_PRAGMA, x.Opcode());
            Xobject x0 = x.getArg(0);
            assertNotNull(x0);
            assertEquals(Xcode.STRING, x0.Opcode());
            assertEquals("SYN_POSTFIX", x0.getString());
            Xobject x1 = x.getArg(1);
            assertNotNull(x1);
            assertEquals(Xcode.STRING, x1.Opcode());
            assertEquals(expectedClause, x1.getString());
            
            if(expectedArg != null) {
                Xobject x2 = x.getArg(2);
                Xobject x20 = x2.getArg(0);
                Xobject x200 = x20.getArg(0);
                assertEquals(Xcode.STRING, x200.Opcode());
                assertEquals(expectedArg, x200.getString());
            }
        }
    }
}
