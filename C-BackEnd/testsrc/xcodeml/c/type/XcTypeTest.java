/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import xcodeml.c.decompile.XcConstObj;
import xcodeml.c.type.XcArrayType;
import xcodeml.c.type.XcBaseTypeEnum;
import xcodeml.c.type.XcComplexType;
import xcodeml.c.type.XcDecimalType;
import xcodeml.c.type.XcEnumType;
import xcodeml.c.type.XcFuncType;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcImagType;
import xcodeml.c.type.XcIntegerType;
import xcodeml.c.type.XcPointerType;
import xcodeml.c.type.XcStructType;
import xcodeml.c.type.XcUnionType;
import xcodeml.c.util.XmcWriter;

public class XcTypeTest
{
    private static String _noLF(String s)
    {
        return s.replaceAll("\\n", "").replaceAll("\\r", "");
    }

    private static void _dump(XmcWriter s)
    {
        System.out.println("----------");
        System.out.println(s);
    }

    @Test
    public void test_appendCode_typeQualifier() throws Exception
    {
        XcIntegerType.Int t = new XcIntegerType.Int();
        t.setIsConst(true);
        t.setIsVolatile(true);
        t.setIsRestrict(true);

        XmcWriter w = new XmcWriter();
        t.appendCode(w, "a");
        _dump(w);
        assertEquals("const volatile restrict int a", w.toString());
    }

    @Test
    public void test_appendCode_pointer() throws Exception
    {
        XcIntegerType.Long tb = new XcIntegerType.Long();
        tb.setIsConst(true);
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setIsVolatile(true);
        tp1.setRefType(tb);

        XmcWriter w = new XmcWriter();
        tp1.appendCode(w, "a");
        _dump(w);
        assertEquals("const long * volatile a", w.toString());
    }

    @Test
    public void test_appendCode_pointer2() throws Exception
    {
        XcIntegerType.Short tb = new XcIntegerType.Short();
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setRefType(tb);
        XcPointerType tp2 = new XcPointerType("P2");
        tp2.setIsRestrict(true);
        tp2.setRefType(tp1);

        XmcWriter w = new XmcWriter();
        tp2.appendCode(w, "a");
        _dump(w);
        assertEquals("short * * restrict a", w.toString());
    }

    @Test
    public void test_appendCode_pointer2TypeQual() throws Exception
    {
        XcIntegerType.Short tb = new XcIntegerType.Short();
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setIsVolatile(true);
        tp1.setRefType(tb);
        XcPointerType tp2 = new XcPointerType("P2");
        tp2.setIsRestrict(true);
        tp2.setRefType(tp1);

        XmcWriter w = new XmcWriter();
        tp2.appendCode(w, "a");
        _dump(w);
        assertEquals("short * volatile * restrict a", w.toString());
    }

    @Test
    public void test_appendCode_array() throws Exception
    {
        XcIntegerType.Char tb = new XcIntegerType.Char();
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setArraySize(2);
        ta1.setRefType(tb);

        XmcWriter w = new XmcWriter();
        ta1.appendCode(w, "a");
        _dump(w);
        assertEquals("char a[2]", w.toString());
    }

    @Test
    public void test_appendCode_arraySizeTypeQual() throws Exception
    {
        XcIntegerType.LongLong tb = new XcIntegerType.LongLong();
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setArraySize(2);
        ta1.setIsConst(true);
        ta1.setIsVolatile(true);
        ta1.setIsRestrict(true);
        ta1.setIsStatic(true);
        ta1.setRefType(tb);

        XmcWriter w = new XmcWriter();
        ta1.appendCode(w, "a");
        _dump(w);
        assertEquals("long long a[const volatile restrict static 2]", w.toString());
    }

    @Test
    public void test_appendCode_array2() throws Exception
    {
        XcDecimalType.Float tb = new XcDecimalType.Float();
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setArraySize(2);
        ta1.setRefType(tb);
        XcArrayType ta2 = new XcArrayType("A2");
        ta2.setArraySize(3);
        ta2.setRefType(ta1);

        XmcWriter w = new XmcWriter();
        ta2.appendCode(w, "a");
        _dump(w);
        assertEquals("float a[3][2]", w.toString());
    }

    @Test
    public void test_appendCode_pointerArray() throws Exception
    {
        XcDecimalType.Double tb = new XcDecimalType.Double();
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setRefType(tb);
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setArraySize(2);
        ta1.setRefType(tp1);

        XmcWriter w = new XmcWriter();
        ta1.appendCode(w, "a");
        _dump(w);
        assertEquals("double * a[2]", w.toString());
    }

    @Test
    public void test_appendCode_arrayPointer() throws Exception
    {
        XcDecimalType.LongDouble tb = new XcDecimalType.LongDouble();
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setRefType(tb);
        ta1.setArraySize(2);
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setRefType(ta1);

        XmcWriter w = new XmcWriter();
        tp1.appendCode(w, "a");
        _dump(w);
        assertEquals("long double (* a)[2]", w.toString());
    }

    @Test
    public void test_appendCode_arrayPointerArray() throws Exception
    {
        XcComplexType.FloatComplex tb = new XcComplexType.FloatComplex();
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setRefType(tb);
        ta1.setArraySize(2);
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setRefType(ta1);
        XcArrayType ta2 = new XcArrayType("A2");
        ta2.setRefType(tp1);
        ta2.setArraySize(3);

        XmcWriter w = new XmcWriter();
        ta2.appendCode(w, "a");
        _dump(w);
        assertEquals("float _Complex (* a[3])[2]", w.toString());
    }

    @Test
    public void test_appendCode_func() throws Exception
    {
        XcFuncType tf1 = new XcFuncType("F1");
        XcIntegerType.Bool tb1 = new XcIntegerType.Bool();
        tf1.setRefType(tb1);
        tf1.setIsInline(true);
        XcIdent p1 = new XcIdent("p1");
        XcComplexType.DoubleComplex ptc1 = new XcComplexType.DoubleComplex();
        p1.setType(ptc1);
        tf1.addParam(p1);
        XcIdent p2 = new XcIdent("p2");
        XcComplexType.LongDoubleComplex ptc2 = new XcComplexType.LongDoubleComplex();
        XcPointerType ptp1 = new XcPointerType("PP1");
        ptp1.setRefType(ptc2);
        p2.setType(ptp1);
        tf1.addParam(p2);

        XmcWriter w = new XmcWriter();
        tf1.appendCode(w, "f");
        _dump(w);
        assertEquals("inline _Bool f(double _Complex p1, long double _Complex * p2)", w.toString());
    }

    @Test
    public void test_appendCode_funcPtr() throws Exception
    {
        XcIntegerType.UInt tb1 = new XcIntegerType.UInt();
        XcFuncType tf1 = new XcFuncType("F1");
        tf1.setRefType(tb1);
        tf1.setIsInline(true);
        XcIdent p1 = new XcIdent("p1");
        XcImagType.DoubleImag ptc1 = new XcImagType.DoubleImag();
        p1.setType(ptc1);
        tf1.addParam(p1);
        XcIdent p2 = new XcIdent("p2");
        XcImagType.LongDoubleImag ptc2 = new XcImagType.LongDoubleImag();
        XcPointerType ptp1 = new XcPointerType("PP1");
        ptp1.setRefType(ptc2);
        p2.setType(ptp1);
        tf1.addParam(p2);
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setRefType(tf1);

        XmcWriter w = new XmcWriter();
        tp1.appendCode(w, "f");
        _dump(w);
        assertEquals("unsigned int (* f)(double _Imaginary, long double _Imaginary * )", w.toString());
    }

    @Test
    public void test_appendCode_enum() throws Exception
    {
        XcEnumType te1 = new XcEnumType("E1");
        te1.setTagName("tag");
        XcIdent id1 = new XcIdent("id1");
        te1.addEnumerator(id1);
        XcIdent id2 = new XcIdent("id2");
        XcConstObj.IntConst ic1 = new XcConstObj.IntConst("3", XcBaseTypeEnum.INT);
        id2.setValue(ic1);
        te1.addEnumerator(id2);

        XmcWriter w = new XmcWriter();
        te1.appendBodyCode(w);
        _dump(w);
        assertEquals("enum tag {id1,id2 = 3}", _noLF(w.toString()));
    }

    @Test
    public void test_appendBodyCode_struct() throws Exception
    {
        XcStructType ts1 = new XcStructType("S1");
        ts1.setTagName("tag");
        XcIdent id1 = new XcIdent("id1");
        XcIntegerType.ULong tid1 = new XcIntegerType.ULong();
        id1.setType(tid1);
        ts1.addMember(id1);
        XcIdent id2 = new XcIdent("id2");
        XcIntegerType.UShort tid2 = new XcIntegerType.UShort();
        id2.setType(tid2);
        ts1.addMember(id2);

        XmcWriter w = new XmcWriter();
        ts1.appendBodyCode(w);
        _dump(w);
        assertEquals("struct tag {unsigned long id1;unsigned short id2;}", _noLF(w.toString()));
    }

    @Test
    public void test_appendCode_structPointer() throws Exception
    {
        XcStructType ts1 = new XcStructType("S1");
        ts1.setTagName("tag");
        XcIdent id1 = new XcIdent("id1");
        XcIntegerType.ULong tid1 = new XcIntegerType.ULong();
        id1.setType(tid1);
        ts1.addMember(id1);
        XcIdent id2 = new XcIdent("id2");
        XcIntegerType.UShort tid2 = new XcIntegerType.UShort();
        id2.setType(tid2);
        ts1.addMember(id2);
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setRefType(ts1);

        XmcWriter w = new XmcWriter();
        tp1.appendCode(w, "a");
        _dump(w);
        assertEquals("struct tag * a", _noLF(w.toString()));
    }

    @Test
    public void test_appendBodyCode_union() throws Exception
    {
        XcUnionType tu1 = new XcUnionType("U1");
        tu1.setTagName("tag");
        XcIdent id1 = new XcIdent("id1");
        XcIntegerType.UChar tid1 = new XcIntegerType.UChar();
        id1.setType(tid1);
        tu1.addMember(id1);
        XcIdent id2 = new XcIdent("id2");
        XcIntegerType.ULongLong tid2 = new XcIntegerType.ULongLong();
        id2.setType(tid2);
        tu1.addMember(id2);

        XmcWriter w = new XmcWriter();
        tu1.appendBodyCode(w);
        _dump(w);
        assertEquals("union tag {unsigned char id1;unsigned long long id2;}", _noLF(w.toString()));
    }

    @Test
    public void test_appendBodyCode_unionHasStruct() throws Exception
    {
        XcUnionType tu1 = new XcUnionType("U1");
        tu1.setTagName("tag1");
        XcIdent id1 = new XcIdent("id1");
        XcStructType tid1 = new XcStructType("S1");
        tid1.setTagName("tag2");
        id1.setType(tid1);
        XcIdent im1 = new XcIdent("id1_m1");
        XcIntegerType.UChar tim1 = new XcIntegerType.UChar();
        im1.setType(tim1);
        tu1.addMember(id1);
        XcIdent id2 = new XcIdent("id2");
        XcImagType.FloatImag tid2 = new XcImagType.FloatImag();
        id2.setType(tid2);
        tu1.addMember(id2);

        XmcWriter w = new XmcWriter();
        tu1.appendBodyCode(w);
        _dump(w);
        assertEquals("union tag1 {struct tag2 id1;float _Imaginary id2;}", _noLF(w.toString()));
    }


    @Test
    public void test_appendCode_variableArray() throws Exception
    {
        XcIdent m = new XcIdent("m");
        XcIdent n = new XcIdent("n");

        XcIntegerType.LongLong tb = new XcIntegerType.LongLong();
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setRefType(tb);
        ta1.setArraySizeExpr(m);
        XcArrayType ta2 = new XcArrayType("A2");
        ta2.setRefType(ta1);
        ta2.setArraySizeExpr(n);

        XmcWriter w = new XmcWriter();
        ta2.appendCode(w, "a");
        _dump(w);
        assertEquals("long long a[n][m]", w.toString());
    }

    @Test
    public void test_appendCode_func_with_variable_array_argument() throws Exception
    {
        XcIdent m = new XcIdent("m");
        XcIdent n = new XcIdent("n");

        XcIntegerType.LongLong tb = new XcIntegerType.LongLong();
        XcArrayType ta1 = new XcArrayType("A1");
        ta1.setRefType(tb);
        ta1.setArraySizeExpr(m);
        XcArrayType ta2 = new XcArrayType("A2");
        ta2.setRefType(ta1);
        ta2.setArraySizeExpr(n);

        XcIdent p1 = new XcIdent("p1");
        p1.setType(ta2);

        XcFuncType tf1 = new XcFuncType("F1");

        XcIntegerType.Int ti1 = new XcIntegerType.Int();
        tf1.setRefType(ti1);

        tf1.addParam(p1);

        XmcWriter w = new XmcWriter();

        tf1.setIsPreDecl(false);
        tf1.appendCode(w, "f");
        _dump(w);
        assertEquals("int f(long long p1[n][m])", w.toString());

        w.close();

        w = new XmcWriter();
         tf1.setIsPreDecl(true);
        tf1.appendCode(w, "f");
        _dump(w);
        assertEquals("int f(long long p1[][*])", w.toString());
    }

    @Test
    public void test_appendCode_basicTypePointer() throws Exception
    {
        XcIntegerType.Int ti1 = new XcIntegerType.Int();

        XcBasicType tb = new XcBasicType("B1", ti1);
        
        tb.setIsConst(true);
        XcPointerType tp1 = new XcPointerType("P1");
        tp1.setRefType(tb);
        assertTrue(tp1 != null);

        XmcWriter w = new XmcWriter();
        tp1.appendCode(w, "a");
        _dump(w);
        assertEquals("const int * a", w.toString());
    }
    
    @Test
    public void test_appendCode_gccAttribute() throws Exception
    {
        XcGccAttributeList attrs = new XcGccAttributeList();
        assertTrue(attrs != null);
        
        attrs.addAttr("aligned(8)");
        attrs.addAttr("aligned(16)");

        XmcWriter w = new XmcWriter();
        attrs.appendCode(w);
        _dump(w);
        assertEquals("__attribute__((aligned(8),aligned(16))) ", w.toString());
    }

    @Test
    public void test_appendCode_gccAttribute_Int() throws Exception
    {
        XcIntegerType.Int ti1 = new XcIntegerType.Int();

        XcBasicType type = new XcBasicType("B1", ti1);
        assertTrue(type != null);

        XcGccAttributeList attrs = new XcGccAttributeList();
        assertTrue(attrs != null);

        attrs.addAttr("unused");
        attrs.addAttr("aligned(8)");

        type.setGccAttribute(attrs);

        XmcWriter w = new XmcWriter();
        type.appendCode(w, "s");
        _dump(w);
        assertEquals("__attribute__((unused,aligned(8))) int s", w.toString());
    }

    @Test
    public void test_appendCode_gccAttribute_pointer_int() throws Exception
    {
        XcIntegerType.Int ti1 = new XcIntegerType.Int();

        XcBasicType bt = new XcBasicType("B1", ti1);
        assertTrue(bt != null);

        XcGccAttributeList intAttrs = new XcGccAttributeList();
        assertTrue(intAttrs != null);

        intAttrs.addAttr("aligned(8)");
        bt.setGccAttribute(intAttrs);

        XcPointerType pt = new XcPointerType("P1");
        pt.setRefType(bt);

        XcGccAttributeList pointerAttrs = new XcGccAttributeList();
        pointerAttrs.addAttr("aligned(32)");
        pt.setGccAttribute(pointerAttrs);

        assertTrue(pt.getGccAttribute() != null);

        XmcWriter w = new XmcWriter();
        pt.appendCode(w, "p");
        _dump(w);

        // assertEquals("__attribute__((aligned(8))) int * __attribute__((aligned(32))) p", w.toString());
    }

    @Test
    public void test_appendCode_gccAttribute_function_type() throws Exception
    {
        XcFuncType tf1 = new XcFuncType("F1");
        XcIntegerType.Bool tb1 = new XcIntegerType.Bool();
        tf1.setRefType(tb1);
        tf1.setIsInline(true);
        XcIdent p1 = new XcIdent("a");
        XcComplexType.DoubleComplex ptc1 = new XcComplexType.DoubleComplex();
        p1.setType(ptc1);
        tf1.addParam(p1);

        XcGccAttributeList attrs1 = new XcGccAttributeList();
        assertTrue(attrs1 != null);
        attrs1.addAttr("noreturn");
        tf1.setGccAttribute(attrs1);

        XmcWriter w = new XmcWriter();
        tf1.appendCode(w, "f");
        _dump(w);
        // assertEquals("inline _Bool f(double _Complex p1, long double _Complex * p2)", w.toString());
    }

    @Test
    public void test_appendCode_gccAttribute_struct_type() throws Exception
    {
        XcStructType ts1 = new XcStructType("S1");
        ts1.setTagName("tag");
        XcIdent id1 = new XcIdent("id1");
        XcIntegerType.ULong tid1 = new XcIntegerType.ULong();
        id1.setType(tid1);
        ts1.addMember(id1);
        XcIdent id2 = new XcIdent("id2");
        XcIntegerType.UShort tid2 = new XcIntegerType.UShort();
        id2.setType(tid2);
        ts1.addMember(id2);

        XcGccAttributeList attrs1 = new XcGccAttributeList();
        assertTrue(attrs1 != null);
        attrs1.addAttr("packed");
        ts1.setGccAttribute(attrs1);

        XmcWriter w = new XmcWriter();
        ts1.appendBodyCode(w);
        _dump(w);
        assertEquals("struct tag {unsigned long id1;unsigned short id2;} __attribute__((packed)) ", _noLF(w.toString()));
    }

    @Test
    public void test_appendCode_gccAttribute_union_type() throws Exception
    {
        XcUnionType tu1 = new XcUnionType("U1");
        tu1.setTagName("tag");
        XcIdent id1 = new XcIdent("id1");
        XcIntegerType.UChar tid1 = new XcIntegerType.UChar();
        id1.setType(tid1);
        tu1.addMember(id1);
        XcIdent id2 = new XcIdent("id2");
        XcIntegerType.ULongLong tid2 = new XcIntegerType.ULongLong();
        id2.setType(tid2);
        tu1.addMember(id2);

        XcGccAttributeList attrs1 = new XcGccAttributeList();
        assertTrue(attrs1 != null);
        attrs1.addAttr("packed");
        tu1.setGccAttribute(attrs1);

        XmcWriter w = new XmcWriter();
        tu1.appendBodyCode(w);
        _dump(w);
        //assertEquals("union tag {unsigned char id1;unsigned long long id2;}", _noLF(w.toString()));
    }

    @Test
    public void test_appendCode_gccAttribute_enum_type() throws Exception
    {
        XcEnumType te1 = new XcEnumType("E1");
        te1.setTagName("tag");
        XcIdent id1 = new XcIdent("id1");
        te1.addEnumerator(id1);
        XcIdent id2 = new XcIdent("id2");
        XcConstObj.IntConst ic1 = new XcConstObj.IntConst("3", XcBaseTypeEnum.INT);
        id2.setValue(ic1);
        te1.addEnumerator(id2);

        XcGccAttributeList attrs1 = new XcGccAttributeList();
        assertTrue(attrs1 != null);
        attrs1.addAttr("packed");
        te1.setGccAttribute(attrs1);

        XmcWriter w = new XmcWriter();
        te1.appendBodyCode(w);
        _dump(w);
        //assertEquals("enum tag {id1,id2 = 3}", _noLF(w.toString()));
    }
}
