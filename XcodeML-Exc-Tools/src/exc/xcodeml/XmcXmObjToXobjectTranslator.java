/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;

import java.util.Stack;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import exc.object.*;
import exc.util.XobjectRecursiveVisitor;
import exc.util.XobjectVisitor;
import xcodeml.IXobject;
import xcodeml.XmException;
import xcodeml.XmObj;
import xcodeml.binding.IRNode;
import xcodeml.binding.IXbLineNo;
import xcodeml.binding.IXbTypedExpr;
import xcodeml.c.binding.*;
import xcodeml.c.binding.gen.*;
import xcodeml.c.decompile.XcConstObj;
import xcodeml.c.util.XmcBindingUtil;
import static xcodeml.util.XmLog.warning;
import static xcodeml.util.XmStringUtil.getAsCInt;
import static xcodeml.util.XmStringUtil.getAsCLong;
import static xcodeml.util.XmStringUtil.getAsBool;
import static xcodeml.util.XmStringUtil.trim;
import xcodeml.util.XmBindingException;
import xcodeml.util.XmLog;
import xcodeml.util.XmStringUtil;
import xcodeml.util.XmXmObjToXobjectTranslator;

/**
 * Visitor for XcodeML/C to Xcode translation.
 */
public class XmcXmObjToXobjectTranslator extends RVisitorBase
    implements XmXmObjToXobjectTranslator
{
    private Xobject _xobj;
    
    private XobjectFile _xobjFile;
    
    private XbcTypeTable _typeTable;

    private static final boolean debug_pre_transform = false;

    enum PragmaScope
    {
        Compexpr, Nestedfunc,
    }

    private static Stack<PragmaScope> _pScopeStack = new Stack<PragmaScope>();
    
    public XmcXmObjToXobjectTranslator()
    {
    }

    private XmcXmObjToXobjectTranslator(XobjectFile xobjFile, XbcTypeTable typeTable)
    {
        _xobjFile = xobjFile;
        _typeTable = typeTable;
    }

    @Override
    public IXobject translate(XmObj xmobj) throws XmException
    {
        try {
            ((IRVisitable)xmobj).enter(this);
            parsePragma();
            getIXobject().setParentRecursively(null);
            return getIXobject();
        } catch(CompileException e) {
            if(e.getXmObj() != null)
                XmLog.error(e.getXmObj(), e);
            else if(e.getIXobject() != null)
                XmLog.error(e.getIXobject().getLineNo(), e);
            else
                XmLog.error(e.getMessage());
            return null;
        }
    }

    private boolean isPragmaAvailable()
    {
        if(_pScopeStack.empty())
            return true;

        if((_pScopeStack.peek() == PragmaScope.Compexpr) ||
           (_pScopeStack.size() >= 2)) {
            return false;
        } else {
            return true;
        }
    }

    private void parsePragma()
    {
        if(_xobjFile == null)
            return;
        
        XobjectVisitor xvisitor = new ParsePragmaVisitor();
        
        if(debug_pre_transform) {
            Writer writer = new java.io.OutputStreamWriter(System.out);
            _xobjFile.Output(writer);
            try { writer.flush(); } catch(Exception e) {}
        }
        
        for(XobjectDef def : _xobjFile) {
            def.enter(xvisitor);
        }
    }
    
    /**
     * transform AST by PragmaSyntax of pragma object.
     */
    class ParsePragmaVisitor extends XobjectRecursiveVisitor
    {
        PragmaParser parser;
        
        ParsePragmaVisitor()
        {
            parser = new PragmaParser(getXobjectFile());
        }
        
        @Override
        public boolean enter(Xobject vv)
        {
            if(!(vv instanceof XobjList))
                return true;

            XobjList v = (XobjList)vv;
            boolean isEnvPushed = false;

            switch(v.Opcode()) {
            case COMPOUND_STATEMENT: {
                    XobjList identList = (XobjList)v.getArg(0);
                    parser.pushEnv(identList);
                    isEnvPushed = true;
                }
                break;
            case FUNCTION_DEFINITION: {
                    XobjList identList = (XobjList)v.getArg(1);
                    parser.pushEnv(identList);
                    isEnvPushed = true;
                }
                break;
            default:
                break;
            }
            
            for(XobjArgs a = v.getArgs(); a != null; a = a.nextArgs()) {
                Xobject x = a.getArg();
                
                if(x == null || !x.isPragma() || x.Opcode() == Xcode.PRAGMA_LINE ||
                    x.isParsed())
                    continue;

                PragmaSyntax syn = PragmaSyntax.valueOrNullOf(x.getArg(0).getString());
                if(syn == null)
                    continue;
                x.removeFirstArgs();
                
                XobjArgs nextArg = null;
                
                if(syn == PragmaSyntax.SYN_PREFIX &&
                    (nextArg = a.nextArgs()) != null) {
                    if(nextArg.getArg().isPragma()) {
                        // complete pragma target region
                        //   p:pragma, s:statement
                        //   p p p s s => p { p p s } s
                        //     ^nextArg     ^nextArg
                        XobjList body = Xcons.List();
                        XobjArgs aa = nextArg;
                        while(aa != null && aa.getArg() != null &&
                            aa.getArg().isPragma()) {
                            body.add(aa.getArg());
                            aa = aa.nextArgs();
                        }
                        if(aa != null && aa.getArg() != null) {
                            body.add(aa.getArg());
                        }
                        nextArg.setArg(Xcons.CompoundStatement(body));
                        nextArg.setNext(aa != null ? aa.nextArgs() : null);
                    }
                    
                    // remove nextArg
                    a.setNext(nextArg.nextArgs());
                    x.add(nextArg.getArg());
                }
                
                try {
                    x = parser.parse(x);
                    x.setIsParsed(true);
                } catch(XmException e) {
                    throw new CompileException(x, e);
                }
                a.setArg(x);
            }
            
            for(Xobject a : v) {
                if(a != null && !a.enter(this))
                    return false;
            }

            if(isEnvPushed) {
                parser.popEnv();
            }

            return true;
        }
    }

    class CompileException extends RuntimeException
    {
        private static final long serialVersionUID = -6715840674962886606L;
        
        private XmObj _xmobj;
        
        private IXobject _ixobj;
        
        CompileException(XmObj xmobj, String msg)
        {
            super(msg);
            _xmobj = xmobj;
        }
        
        CompileException(IXobject ixobj, Exception e)
        {
            super(e);
            _ixobj = ixobj;
        }
        
        XmObj getXmObj()
        {
            return _xmobj;
        }

        IXobject getIXobject()
        {
            return _ixobj;
        }
    }
    
    public IXobject getIXobject()
    {
        if(_xobj != null) {
            return _xobj;
        }
        
        return _xobjFile;
    }
    
    public Xobject getXobject()
    {
        return _xobj;
    }
    
    public XobjectFile getXobjectFile()
    {
        return _xobjFile;
    }
    
    /* -------------------------------------------------
     * utility methods start
     */
    
    private abstract class DefaultObj extends XmObj implements IRVisitable
    {
        @Override
        public void makeTextElement(StringBuffer buffer)
        {
        }

        @Override
        public void makeTextElement(Writer buffer) throws IOException
        {
        }

        @Override
        public void setup(Reader reader) throws IOException, SAXException,
            ParserConfigurationException
        {
        }

        @Override
        public boolean enter(IRVisitor visitor)
        {
            return false;
        }

        @Override
        public void leave(IRVisitor visitor)
        {
        }

        @Override
        public IRNode rGetParentRNode()
        {
            return null;
        }

        @Override
        public IRNode[] rGetRNodes()
        {
            return null;
        }

        @Override
        public void rSetParentRNode(IRNode parent)
        {
        }
    }
    
    private class SymbolObj extends DefaultObj
    {
        Xcode _code;
        private String _name;
        private Xtype _type;
        private VarScope _scope;
        
        SymbolObj(Xcode code, IXbTypedExpr xmobj, String name, String scope)
        {
            _code = code;
            _name = name;
            if(xmobj != null)
                _type = getType(xmobj);
            if(scope != null)
                _scope = VarScope.get(scope);
        }

        SymbolObj(Xcode code, IXbTypedExpr xmobj, String name)
        {
            this(code, xmobj, name, null);
        }
        
        SymbolObj(IXbTypedExpr xmobj, String name)
        {
            this(Xcode.IDENT, xmobj, name);
        }
        
        SymbolObj(Xcode code, String name)
        {
            this(code, null, name, null);
        }
        
        SymbolObj(String name)
        {
            this((IXbTypedExpr)null, name);
        }
        
        SymbolObj(XbcName name)
        {
            this(name.getContent());
        }
        
        @Override
        public boolean enter(IRVisitor visitor)
        {
            ((XmcXmObjToXobjectTranslator)visitor)._xobj =
                Xcons.Symbol(_code, _type, _name, _scope);
            return true;
        }
    }
    
    private class TypeNameObj extends DefaultObj
    {
        private String _tid;
        
        TypeNameObj(String tid)
        {
            _tid = tid;
        }
        
        @Override
        public boolean enter(IRVisitor visitor)
        {
            ((XmcXmObjToXobjectTranslator)visitor)._xobj =
                Xcons.List(Xcode.TYPE_NAME, getType(_tid));
            return true;
        }
    }
    
    private class StringObj extends DefaultObj
    {
        private Xcode _code;
        private String _value;
        
        StringObj(Xcode code, String value)
        {
            _code = code;
            _value = value;
        }
        
        StringObj(String value)
        {
            this(Xcode.STRING_CONSTANT, value);
        }
        
        @Override
        public boolean enter(IRVisitor visitor)
        {
            ((XmcXmObjToXobjectTranslator)visitor)._xobj =
                new XobjString(_code, _value);
            return true;
        }
    }

    class IntFlagObj extends DefaultObj
    {
        private XmObj _xmobj;
        private String _boolStr;
        
        IntFlagObj(XmObj xmobj, String boolStr)
        {
            _xmobj = xmobj;
            _boolStr = boolStr;
        }
        
        Xobject toXobject()
        {
            boolean b = toBool(_xmobj, _boolStr);
            return Xcons.Int(Xcode.INT_CONSTANT, (b ? 1 : 0));
        }
        
        @Override
        public boolean enter(IRVisitor visitor)
        {
            ((XmcXmObjToXobjectTranslator)visitor)._xobj = toXobject();
            return true;
        }
    }
    
    private Xtype getType(String tid)
    {
        if(tid == null) {
            throw new XmBindingException(null, "type id is null");
        }
        return getXobjectFile().findType(tid);
    }
    
    private Xtype getType(IXbTypedExpr xmobj)
    {
        return getXobjectFile().findType(xmobj.getType());
    }
    
    private boolean enterChild(IRVisitable xmobj)
    {
        if(xmobj != null)
            xmobj.enter(this);
        return true;
    }
    
    private boolean enterAsXobject(IRVisitable xmobj)
    {
        boolean r = xmobj.enter(this);
        setCommonAttributes((XmObj)xmobj);
        return r;
    }
    
    private Xobject toXobject(IRVisitable xmobj)
    {
        if(xmobj == null)
            return null;
        
        XmcXmObjToXobjectTranslator t = new XmcXmObjToXobjectTranslator(_xobjFile, _typeTable);
        if(!xmobj.enter(t))
            return null;
        return t.getXobject();
    }
    
    private XobjList toXobjList(Xcode code, Xtype type, IRVisitable ... xmobjs)
    {
        if(xmobjs == null)
            return null;
        
        XobjList objList = Xcons.List(code, type);
        
        for(IRVisitable v : xmobjs) {
            if(v != null) {
                Xobject obj = toXobject(v);
                objList.add(obj); // add even if obj is null
            } else {
                objList.add(null);
            }
        }
        
        return objList;
    }

    private boolean enterAsXobjList(XmObj xmobj, Xcode code, IRVisitable ... childXmobjs)
    {
        Xtype type = null;
        if(xmobj instanceof IXbTypedExpr &&
            (!(xmobj instanceof XbcBuiltinOp) ||
                (!toBool(xmobj, ((XbcBuiltinOp)xmobj).getIsId()) &&
                 !toBool(xmobj, ((XbcBuiltinOp)xmobj).getIsAddrOf())))) {
            type = getType(((IXbTypedExpr)xmobj).getType());
        }
        if(childXmobjs != null)
            _xobj = toXobjList(code, type, childXmobjs);
        else
            _xobj = toXobjList(code, type);
        setCommonAttributes(xmobj);
        return true;
    }
    
    private static boolean toBool(XmObj xmobj, String boolStr)
    {
        return (boolStr == null || boolStr.length() == 0) ? false : getAsBool(xmobj, boolStr);
    }
    
    private static int getTypeQualFlags(IXbcType type)
    {
        int tqConst = toBool((XmObj)type, type.getIsConst()) ? Xtype.TQ_CONST : 0;
        int tqRestrict = toBool((XmObj)type, type.getIsRestrict()) ? Xtype.TQ_RESTRICT : 0;
        int tqVolatile = toBool((XmObj)type, type.getIsVolatile()) ? Xtype.TQ_VOLATILE : 0;
        int tqInline = ((type instanceof XbcFunctionType) &&
            toBool((XmObj)type, ((XbcFunctionType)type).getIsInline())) ? Xtype.TQ_INLINE : 0;
        int tqFuncStatic = ((type instanceof XbcFunctionType) &&
            toBool((XmObj)type, ((XbcFunctionType)type).getIsStatic())) ? Xtype.TQ_FUNC_STATIC : 0;
        
        return tqConst | tqRestrict | tqVolatile | tqInline | tqFuncStatic;
    }
    
    private void setCommonAttributes(XmObj xmobj)
    {
        if(xmobj instanceof IXbLineNo) {
            IXbLineNo i = (IXbLineNo)xmobj;
            if(i.getLineno() != null) {
                _xobj.setLineNo(new LineNo(
                    i.getFile(), Integer.parseInt(i.getLineno())));
            }
        }
        
        if(xmobj instanceof IXbcAnnotation) {
            IXbcAnnotation i = (IXbcAnnotation)xmobj;
            _xobj.setIsGccSyntax(toBool(xmobj, i.getIsGccSyntax()));
            _xobj.setIsSyntaxModified(toBool(xmobj, i.getIsModified()));
        }
        
        if(xmobj instanceof IXbcHasGccExtension) {
            IXbcHasGccExtension i = (IXbcHasGccExtension)xmobj;
            _xobj.setIsGccExtension(toBool(xmobj, i.getIsGccExtension()));
        }
    }
    
    private void setCommonAttributes(Xobject xobj, XmObj xmobj)
    {
        if(xmobj instanceof IXbLineNo) {
            IXbLineNo i = (IXbLineNo)xmobj;
            if(i.getLineno() != null) {
                xobj.setLineNo(new LineNo(i.getFile(), Integer.parseInt(i.getLineno())));
            }
        }
        
        if(xmobj instanceof IXbcAnnotation) {
            IXbcAnnotation i = (IXbcAnnotation)xmobj;
            xobj.setIsGccSyntax(toBool(xmobj, i.getIsGccSyntax()));
            xobj.setIsSyntaxModified(toBool(xmobj, i.getIsModified()));
        }
        
        if(xmobj instanceof IXbcHasGccExtension) {
            IXbcHasGccExtension i = (IXbcHasGccExtension)xmobj;
            xobj.setIsGccExtension(toBool(xmobj, i.getIsGccExtension()));
        }
    }
    
    private boolean enterUnaryExpr(Xcode code, IXbcUnaryExpr xmobj)
    {
        return enterAsXobjList((XmObj)xmobj, code, xmobj.getExpressions());
    }
    
    private boolean enterBinaryExpr(Xcode code, IXbcBinaryExpr xmobj)
    {
        return enterAsXobjList((XmObj)xmobj, code,
            xmobj.getExpressions1(), xmobj.getExpressions2());
    }
    
    private boolean enterMember(Xcode code, IXbcMember xmobj)
    {
        return enterAsXobjList((XmObj)xmobj, code,
            xmobj.getExpressions(),
            new SymbolObj(xmobj, xmobj.getMember()));
    }
    
    private boolean enterVar(Xcode code, IXbcVar xmobj)
    {
        return enterAsXobject(new SymbolObj(
            code, xmobj, xmobj.getContent(), xmobj.getScope()));
    }
    
    private boolean enterSymbolAddr(Xcode code, IXbcSymbolAddr xmobj, String scope)
    {
        _xobj = Xcons.Symbol(code, getType(xmobj), xmobj.getContent(),
            ((scope != null) ? VarScope.get(scope) : null));
        setCommonAttributes((XmObj)xmobj);
        return true;
    }
    
    private boolean enterNoArgObj(Xcode code, XmObj xmobj)
    {
        return enterAsXobjList(xmobj, code, (IRVisitable[])null);
    }
    
    private boolean enterOneArgObj(Xcode code, XmObj xmobj, IRVisitable child)
    {
        return enterAsXobjList(xmobj, code, child);
    }

    private boolean enterSizeOrAlignExpr(Xcode code, IXbcSizeOrAlignExpr xmobj, String exprType)
    {
        String argType = xmobj.getExprOrType().getType();
        Xtype type = getType(argType);

        for(IXbcTypesChoice tc : _typeTable.getTypes()) {
            if((argType.equals(tc.getType())) &&
                (tc instanceof XbcArrayType) &&
                (((XbcArrayType)tc).getArraySize2() != null)) {
                
                type = Xtype.voidPtrType;
                break;
            }
        }

        XobjList child = toXobjList(Xcode.TYPE_NAME, type, new IRVisitable[0]);
        _xobj = toXobjList(code, getType(exprType), new IRVisitable[0]);
        _xobj.add(child);

        return true;
    }
    
    /* utility methods end
     * -------------------------------------------------
     */
    
    @Override
    public boolean enter(XbcArguments xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST, xmobj.getExpressions());
    }
    
    @Override
    public boolean enter(XbcArrayAddr xmobj)
    {
        _xobj = Xcons.Symbol(Xcode.ARRAY_ADDR, getType(xmobj),
            xmobj.getContent(), VarScope.get(xmobj.getScope()));
        setCommonAttributes(xmobj);
        return true;
    }

    private Xobject toXobject(XbcArrayAddr xmobj)
    {
        Xobject xobj = Xcons.Symbol(Xcode.ARRAY_ADDR, getType(xmobj),
                                    xmobj.getContent(), VarScope.get(xmobj.getScope()));
        setCommonAttributes(xobj, xmobj);
        return xobj;
    }

    @Override
    public boolean enter(XbcArrayRef xmobj)
    {
        _xobj = Xcons.List(Xcode.ARRAY_REF, getType(xmobj),
                           toXobject(xmobj.getArrayAddr()),
                           toXobjList(Xcode.LIST, getType(xmobj), xmobj.getExpressions()));
        return true;
    }

    @Override
    public boolean enter(XbcArraySize xmobj)
    {
        return enterAsXobject(xmobj.getExpressions());
    }
    
    @Override
    public boolean enter(XbcArrayType xmobj)
    {
        String size1 = xmobj.getArraySize1();
        long size = -1;
        Xobject sizeExpr = null;
        
        if("*".equals(size1)) {
            sizeExpr = toXobject(xmobj.getArraySize2());
        } else if(size1 != null) {
            size = getAsCLong(xmobj, size1);
        }
        
        Xtype refType = getType(xmobj.getElementType());
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        Xtype type = null;
        
        type = new ArrayType(xmobj.getType(), refType,
            getTypeQualFlags(xmobj), size, sizeExpr, gccAttrs);
        
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcAsgBitAndExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_BIT_AND_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgBitOrExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_BIT_OR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgBitXorExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_BIT_XOR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgDivExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_DIV_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgLshiftExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_LSHIFT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgMinusExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_MINUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgModExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_MOD_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgMulExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_MUL_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgPlusExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_PLUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAsgRshiftExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASG_RSHIFT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcAssignExpr xmobj)
    {
        return enterBinaryExpr(Xcode.ASSIGN_EXPR, xmobj);
    }
    
    @Override
    public boolean enter(XbcBasicType xmobj)
    {
        String tid = xmobj.getType();
        Xtype type;
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        BasicType.TypeInfo ti = BasicType.getTypeInfoByCName(xmobj.getName());
        int tq = getTypeQualFlags(xmobj);
        
        if(ti == null) {
            // inherited type
            Xtype ref = getType(xmobj.getName());
            type = ref.inherit(tid);
            type.setTypeQualFlags(tq);
            type.setGccAttributes(gccAttrs);
        } else {
            type = new BasicType(ti.type.getBasicType(), tid, tq, gccAttrs, null, null);
        }
        
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcBitAndExpr xmobj)
    {
        return enterBinaryExpr(Xcode.BIT_AND_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcBitField xmobj)
    {
        return enterAsXobject(xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcBitNotExpr xmobj)
    {
        return enterUnaryExpr(Xcode.BIT_NOT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcBitOrExpr xmobj)
    {
        return enterBinaryExpr(Xcode.BIT_OR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcBitXorExpr xmobj)
    {
        return enterBinaryExpr(Xcode.BIT_XOR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcBody xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST, xmobj.getStatements());
    }

    @Override
    public boolean enter(XbcBreakStatement xmobj)
    {
        return enterNoArgObj(Xcode.BREAK_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbcBuiltinOp xmobj)
    {
        // (CODE name is_id is_addr content...)
        boolean r = enterAsXobjList(xmobj, Xcode.BUILTIN_OP,
            new StringObj(Xcode.IDENT, xmobj.getName()),
            new IntFlagObj(xmobj, xmobj.getIsId()),
            new IntFlagObj(xmobj, xmobj.getIsAddrOf()));

        XobjList args = toXobjList(Xcode.LIST, null, xmobj.getContent());
        if(args != null)
            _xobj.add(args);
        
        return r;
    }

    @Override
    public boolean enter(XbcCaseLabel xmobj)
    {
        return enterOneArgObj(Xcode.CASE_LABEL, xmobj, xmobj.getValue());
    }

    @Override
    public boolean enter(XbcCastExpr xmobj)
    {
        return enterOneArgObj(Xcode.CAST_EXPR, xmobj, xmobj.getContent());
    }

    @Override
    public boolean enter(XbcXmpDescOf xmobj)
    {
        return enterAsXobjList((XmObj)xmobj, Xcode.XMP_DESC_OF, 
			       xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcCoArrayAssignExpr xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.CO_ARRAY_ASSIGN_EXPR,
            xmobj.getExpressions1(),
            xmobj.getExpressions2());
    }

    @Override
    public boolean enter(XbcCoArrayRef xmobj)
    {
        //boolean r = enterAsXobjList(xmobj, Xcode.CO_ARRAY_REF,
	//new SymbolObj(xmobj.getName()));
        boolean r = enterAsXobjList(xmobj, Xcode.CO_ARRAY_REF,
				    xmobj.getContent());
        Xobject exprs = Xcons.List();
        for(IRVisitable v : xmobj.getExpressions()) {
            exprs.add(toXobject(v));
        }
        _xobj.add(exprs);
        _xobj.setScope(VarScope.get(xmobj.getScope()));
        return r;
    }

    @Override
    public boolean enter(XbcCoArrayType xmobj)
    {
        String size1 = xmobj.getArraySize1();
        int size = -1;
        Xobject sizeExpr = null;
        
        if("*".equals(size1)) {
            sizeExpr = toXobject(xmobj.getArraySize2());
        } else if(size1 != null) {
            size = getAsCInt(xmobj, size1);
        }
        
        Xtype refType = getType(xmobj.getElementType());
        Xtype type = new XmpCoArrayType(xmobj.getType(), refType, 0, size, sizeExpr);
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcCommaExpr xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.COMMA_EXPR, xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcCompoundStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.COMPOUND_STATEMENT,
            xmobj.getSymbols(),
            xmobj.getDeclarations(),
            xmobj.getBody());
    }

    @Override
    public boolean enter(XbcCompoundValue xmobj)
    {
        // XbcCompoundValue is non-top level 'value' element.
        return enterAsXobjList(xmobj, Xcode.LIST,
            xmobj.getCompoundLiteral());
    }

    @Override
    public boolean enter(XbcCompoundValueAddrExpr xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.COMPOUND_VALUE_ADDR,
            xmobj.getValue());
    }

    @Override
    public boolean enter(XbcCompoundValueExpr xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.COMPOUND_VALUE,
            xmobj.getValue());
    }

    @Override
    public boolean enter(XbcCondExpr xmobj)
    {
        _xobj = Xcons.List(Xcode.CONDITIONAL_EXPR,
            getType(xmobj),
            toXobject(xmobj.getExpressions1()),
            Xcons.List(Xcode.LIST,
                toXobject(xmobj.getExpressions2()),
                toXobject(xmobj.getExpressions3())));
        setCommonAttributes(xmobj);
        return true;
    }

    @Override
    public boolean enter(XbcCondition xmobj)
    {
        return enterAsXobject(xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcContinueStatement xmobj)
    {
        return enterNoArgObj(Xcode.CONTINUE_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbcDeclarations xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST, xmobj.getContent());
    }

    @Override
    public boolean enter(XbcDefaultLabel xmobj)
    {
        return enterNoArgObj(Xcode.DEFAULT_LABEL, xmobj);
    }

    @Override
    public boolean enter(XbcDesignatedValue xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.DESIGNATED_VALUE,
            xmobj.getContent(),
            new SymbolObj(xmobj.getMember()));
    }

    @Override
    public boolean enter(XbcDivExpr xmobj)
    {
        return enterBinaryExpr(Xcode.DIV_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcDoStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.DO_STATEMENT,
            xmobj.getBody(),
            xmobj.getCondition());
    }

    @Override
    public boolean enter(XbcElse xmobj)
    {
        return enterChild(xmobj.getStatements());
    }

    @Override
    public boolean enter(XbcEnumType xmobj)
    {
        XobjList identList = (XobjList)toXobject(xmobj.getSymbols());
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        Xtype type = new EnumType(xmobj.getType(), identList,
            getTypeQualFlags(xmobj), gccAttrs);
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcExprStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.EXPR_STATEMENT,
            xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcFloatConstant xmobj)
    {
        _xobj = Xcons.Float(Xcode.FLOAT_CONSTANT, getType(xmobj), xmobj.getContent());
        return true;
    }

    @Override
    public boolean enter(XbcForStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.FOR_STATEMENT,
            xmobj.getInit(), xmobj.getCondition(), xmobj.getIter(), xmobj.getBody());
    }

    @Override
    public boolean enter(XbcFuncAddr xmobj)
    {
        return enterAsXobject(
            new SymbolObj(Xcode.FUNC_ADDR, xmobj, xmobj.getContent()));
    }

    @Override
    public boolean enter(XbcFunction xmobj)
    {
        return enterChild(xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcFunctionCall xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.FUNCTION_CALL,
            xmobj.getFunction(),
            xmobj.getArguments());
    }

    @Override
    public boolean enter(XbcFunctionDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.FUNCTION_DECL,
            new SymbolObj(xmobj.getName()),
            xmobj.getGccAsm());
    }

    @Override
    public boolean enter(XbcFunctionDefinition xmobj)
    {
        _pScopeStack.push(PragmaScope.Nestedfunc);

        if(!enterAsXobjList(xmobj, Xcode.FUNCTION_DEFINITION,
            new SymbolObj(xmobj.getName()),
            xmobj.getSymbols(),
            xmobj.getParams(),
            xmobj.getBody(),
            xmobj.getGccAttributes()))
            return false;

        _pScopeStack.pop();
        
        // change body to COMPOUND_STATTEMENT.
        // body of FUNCTION_DEFINITION must be COMPOUND_STATEMENT.
        XobjArgs arg = _xobj.getArgs();
        XobjArgs argBody = arg.nextArgs().nextArgs().nextArgs();
        Xobject stmts = argBody.getArg();
        if(stmts != null) {
            switch(stmts.Opcode()) {
            case LIST:
                stmts = Xcons.CompoundStatement(
                    Xcons.IDList(), Xcons.List(), stmts);
                argBody.setArg(stmts);
                break;
            case COMPOUND_STATEMENT:
                break;
            default:
                XmLog.fatal(stmts.toString());
            }
        }
        return true;
    }

    @Override
    public boolean enter(XbcFunctionType xmobj)
    {
        Xtype retType = getType(xmobj.getReturnType());
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        Xobject params = null;
        
        if(xmobj.getParams() != null ) {
            if(xmobj.getParams().getName().length > 0) {
                params = Xcons.List();
                if(xmobj.getParams().getName().length >= 2 ||
                    !"void".equals(xmobj.getParams().getName(0).getType())) {
                    for(XbcName xmname : xmobj.getParams().getName()) {
                        Xtype type = getType(xmname.getType());
                        params.add(Xcons.Symbol(Xcode.IDENT, type, xmname.getContent()));
                    }
                }
            }
            
            if(xmobj.getParams().getEllipsis() != null) {
                if(params == null) {
                    params = Xcons.List();
                }
                params.add(null);
            }
        }
        
        Xtype type = new FunctionType(xmobj.getType(), retType,
            params, getTypeQualFlags(xmobj), false, gccAttrs, null);
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcGccAlignOfExpr xmobj)
    {
        return enterSizeOrAlignExpr(Xcode.GCC_ALIGN_OF_EXPR, xmobj, xmobj.getType());
    }

    @Override
    public boolean enter(XbcGccAsm xmobj)
    {
        return enterOneArgObj(Xcode.GCC_ASM, xmobj, xmobj.getStringConstant());
    }

    @Override
    public boolean enter(XbcGccAsmClobbers xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.GCC_ASM_CLOBBERS, xmobj.getStringConstant());
    }

    @Override
    public boolean enter(XbcGccAsmDefinition xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.GCC_ASM_DEFINITION, xmobj.getStringConstant());
    }

    @Override
    public boolean enter(XbcGccAsmOperand xmobj)
    {
        StringObj matchObj;
        if (xmobj.getMatch() == null) {
          matchObj = null;
        }
        else {
          matchObj = new StringObj(xmobj.getMatch());
        }

        StringObj constraintObj;
        if (xmobj.getConstraint() == null) {
          constraintObj = null;
        }
        else {
          constraintObj = new StringObj(xmobj.getConstraint());
        }

        return enterAsXobjList(xmobj, Xcode.GCC_ASM_OPERAND,
            xmobj.getExpressions(),
            matchObj,
            constraintObj);
    }

    @Override
    public boolean enter(XbcGccAsmOperands xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.GCC_ASM_OPERANDS, xmobj.getGccAsmOperand());
    }

    @Override
    public boolean enter(XbcGccAsmStatement xmobj)
    {
        // (CODE is_volatile string_constant operand1|() operand2|() clobbers|())
        boolean r = enterAsXobjList(xmobj, Xcode.GCC_ASM_STATEMENT,
            new IntFlagObj(xmobj, xmobj.getIsVolatile()),
            xmobj.getStringConstant());
        
        if(xmobj.getGccAsmOperands1() != null) {
            _xobj.add(toXobject(xmobj.getGccAsmOperands1()));
        } else {
            _xobj.add(null);
        }

        if(xmobj.getGccAsmOperands2() != null) {
            _xobj.add(toXobject(xmobj.getGccAsmOperands2()));
        } else {
            _xobj.add(null);
        }
        
        if(xmobj.getGccAsmClobbers() != null) {
            _xobj.add(toXobject(xmobj.getGccAsmClobbers()));
        } else {
            _xobj.add(null);
        }
        
        return r;
    }

    @Override
    public boolean enter(XbcGccAttribute xmobj)
    {
        boolean r = enterAsXobjList(xmobj, Xcode.GCC_ATTRIBUTE,
            new SymbolObj(xmobj.getName()));
        Xobject exprs = Xcons.List();
        for(IRVisitable v : xmobj.getExpressions()) {
            exprs.add(toXobject(v));
        }
        _xobj.add(exprs);
        return r;
    }

    @Override
    public boolean enter(XbcGccAttributes xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.GCC_ATTRIBUTES, xmobj.getGccAttribute());
    }

    @Override
    public boolean enter(XbcGccCompoundExpr xmobj)
    {
        _pScopeStack.push(PragmaScope.Compexpr);
        boolean ret = enterAsXobjList(xmobj, Xcode.GCC_COMPOUND_EXPR,
            xmobj.getCompoundStatement());
        _pScopeStack.pop();
        return ret;
    }

    @Override
    public boolean enter(XbcGccLabelAddr xmobj)
    {
        return enterAsXobject(new SymbolObj(Xcode.GCC_LABEL_ADDR, xmobj.getContent()));
    }

    @Override
    public boolean enter(XbcGccMemberDesignator xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.GCC_MEMBER_DESIGNATOR,
            new TypeNameObj(xmobj.getRef()),
            new SymbolObj(xmobj.getMember()),
            xmobj.getExpressions(),
            xmobj.getGccMemberDesignator());
    }

    @Override
    public boolean enter(XbcGccRangedCaseLabel xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.CASE_LABEL,
            xmobj.getValue1(), xmobj.getValue2());
    }

    @Override
    public boolean enter(XbcGlobalDeclarations xmobj)
    {
        Xobject decls = toXobjList(Xcode.LIST, null, xmobj.getContent());
        if(decls != null)
            getXobjectFile().add(decls);
        return true;
    }
    
    @Override
    public boolean enter(XbcGlobalSymbols xmobj)
    {
        Xobject identList = new XobjList(Xcode.ID_LIST);
        
        for(XbcId xmId : xmobj.getId()) {
            Ident id = (Ident)toXobject(xmId);
            id.setIsGlobal(true);
            identList.add(toXobject(xmId));
        }
        
        _xobj = identList;
        
        return true;
    }

    @Override
    public boolean enter(XbcGotoStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.GOTO_STATEMENT, xmobj.getContent());
    }

    @Override
    public boolean enter(XbcId xmobj)
    {
        String name = xmobj.getName().getContent();
        
        // get type
        String tid = xmobj.getType();
        Xtype type = null;
        
        if(tid == null)
            tid = xmobj.getName().getType();
        
        if(tid != null) {
            type = getType(tid);
        }

        // get storage class
        StorageClass sclass = StorageClass.SNULL;
        String sclassStr = trim(xmobj.getSclass());
        
        if(sclassStr != null) {
            sclass = StorageClass.get(sclassStr);
        }
        
        // get bit field
        String bitFieldStr = xmobj.getBitField1();
        int bitField = 0;
        Xobject bitFieldExpr = null;
        
        if(bitFieldStr != null) {
            if("*".equals(bitFieldStr)) {
                bitFieldExpr = toXobject(xmobj.getBitField2());
            } else {
                bitField = getAsCInt(xmobj, bitFieldStr);
            }
        }

        // get enum member value
        Xobject enumValue = null;
        if(xmobj.getValue() != null) {
            enumValue = toXobject(xmobj.getValue().getContent());
        }
        
        // get gcc attributes
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        
        // get optional flags
        int optionalFlags =
            (toBool(xmobj, xmobj.getIsGccExtension()) ? Xobject.OPT_GCC_EXTENSION : 0) |
            (toBool(xmobj, xmobj.getIsGccThread()) ? Xobject.OPT_GCC_THREAD : 0);
        
        // addr of symbol
        Xobject addr = null;
        if(sclass != null && sclass.canBeAddressed()) {
            switch(type.getKind()) {
            case Xtype.BASIC:
            case Xtype.POINTER:
            case Xtype.FUNCTION:
            case Xtype.ENUM:
            case Xtype.STRUCT:
            case Xtype.UNION: {
                    Xtype ptype = Xtype.Pointer(type, getXobjectFile());
                    addr = Xcons.Symbol(Xcode.VAR_ADDR, ptype,
                        name, VarScope.LOCAL);
                }
                break;
            case Xtype.ARRAY: {
                    Xtype ptype = Xtype.Pointer(type, getXobjectFile());
                    addr = Xcons.Symbol(Xcode.ARRAY_ADDR, ptype,
                        name, VarScope.LOCAL);
                }
                break;
            }
        }
        
        // create ident
        Ident ident = new Ident(name, sclass, type, addr,
            optionalFlags, gccAttrs, bitField, bitFieldExpr, enumValue, null);
        
        _xobj = ident;
        
        
        // declaring
        if(sclass != null && sclass.isVarOrFunc()) {
            ident.setIsDeclared(true);
        }
        
        return true;
    }

    @Override
    public boolean enter(XbcIfStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.IF_STATEMENT,
            xmobj.getCondition(), xmobj.getThen(), xmobj.getElse());
    }

    @Override
    public boolean enter(XbcInit xmobj)
    {
        return enterChild(xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcIntConstant xmobj)
    {
        _xobj = Xcons.Int(Xcode.INT_CONSTANT,
            getType(xmobj),
            XmStringUtil.getAsCInt(xmobj, xmobj.getContent()));
        return true;
    }

    @Override
    public boolean enter(XbcIter xmobj)
    {
        return enterChild(xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcLogAndExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_AND_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogEQExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_EQ_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogGEExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_GE_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogGTExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_GT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogLEExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_LE_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogLTExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_LT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogNEQExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_NEQ_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogNotExpr xmobj)
    {
        return enterUnaryExpr(Xcode.LOG_NOT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLogOrExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_OR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcLonglongConstant xmobj)
    {
        XcConstObj.LongLongConst llConst = XmcBindingUtil.createLongLongConst(xmobj);
        _xobj = Xcons.Long(Xcode.LONGLONG_CONSTANT, getType(xmobj),
            (int)llConst.getHigh(), (int)llConst.getLow());
        return true;
    }

    @Override
    public boolean enter(XbcLshiftExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LSHIFT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcMemberAddr xmobj)
    {
        return enterMember(Xcode.MEMBER_ADDR, xmobj);
    }

    @Override
    public boolean enter(XbcMemberArrayAddr xmobj)
    {
        return enterMember(Xcode.MEMBER_ARRAY_ADDR, xmobj);
    }

    @Override
    public boolean enter(XbcMemberArrayRef xmobj)
    {
        return enterMember(Xcode.MEMBER_ARRAY_REF, xmobj);
    }

    @Override
    public boolean enter(XbcMemberRef xmobj)
    {
        return enterMember(Xcode.MEMBER_REF, xmobj);
    }

    @Override
    public boolean enter(XbcMinusExpr xmobj)
    {
        return enterBinaryExpr(Xcode.MINUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcModExpr xmobj)
    {
        return enterBinaryExpr(Xcode.MOD_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcMoeConstant xmobj)
    {
        _xobj = new XobjString(Xcode.MOE_CONSTANT, getType(xmobj), xmobj.getContent());
        return true;
    }

    @Override
    public boolean enter(XbcMulExpr xmobj)
    {
        return enterBinaryExpr(Xcode.MUL_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcName xmobj)
    {
        _xobj = new XobjString(Xcode.IDENT, xmobj.getContent());
        return true;
    }

    @Override
    public boolean enter(XbcParams xmobj)
    {
        // for functionDefinition/params
        // NOT for functionType/params

        XobjList xobj = Xcons.List();
        for(XbcName xmname : xmobj.getName()) {
            if((xmname.getContent() != null) &&
               (!xmname.getContent().equals("")))
                xobj.add(Xcons.List(Xcode.VAR_DECL,
                    Xcons.Symbol(Xcode.IDENT, xmname.getContent()), null));
        }
        
        if(toBool(xmobj, xmobj.getEllipsis())) {
            xobj.add(null);
        }
        _xobj = xobj;
        
        return true;
    }

    @Override
    public boolean enter(XbcPlusExpr xmobj)
    {
        return enterBinaryExpr(Xcode.PLUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcPointerRef xmobj)
    {
        return enterUnaryExpr(Xcode.POINTER_REF, xmobj);
    }

    @Override
    public boolean enter(XbcPointerType xmobj)
    {
        Xtype refType = getXobjectFile().findType(xmobj.getRef());
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        Xtype type = new PointerType(xmobj.getType(), refType,
            getTypeQualFlags(xmobj), gccAttrs);
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcPostDecrExpr xmobj)
    {
        return enterUnaryExpr(Xcode.POST_DECR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcPostIncrExpr xmobj)
    {
        return enterUnaryExpr(Xcode.POST_INCR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcPragma xmobj)
    {
        PragmaLexer lexer = new CpragmaLexer(this, xmobj);
        PragmaLexer.Result result = null;

        try {
            result = lexer.lex(xmobj.getContent());
        } catch(XmException e) {
            // fatal binding error
            throw new XmBindingException(xmobj, e);
        }
        
        if(result != null) {
            if(result.error_message != null) {
                // pragma syntax/semantic error
                throw new CompileException(xmobj, result.error_message);
            }
        
            _xobj = result.xobject;

            if(isPragmaAvailable() == false) {
                warning(xmobj, "ignored. directive is not allowed here");
                _xobj = null;
            }
        }
        
        if(_xobj == null) {
            _xobj = Xcons.List(Xcode.PRAGMA_LINE,
                new XobjString(Xcode.STRING, xmobj.getContent()));
        } else if(xmobj.getLineno() != null) {
            LineNo ln = new LineNo(xmobj.getFile(),
                Integer.parseInt(xmobj.getLineno()));
            _xobj.setLineNo(ln);
        }
        return true;
    }

    @Override
    public boolean enter(XbcPreDecrExpr xmobj)
    {
        return enterUnaryExpr(Xcode.PRE_DECR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcPreIncrExpr xmobj)
    {
        return enterUnaryExpr(Xcode.PRE_INCR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcReturnStatement xmobj)
    {
        return enterOneArgObj(Xcode.RETURN_STATEMENT, xmobj, xmobj.getExpressions());
    }

    @Override
    public boolean enter(XbcRshiftExpr xmobj)
    {
        return enterBinaryExpr(Xcode.RSHIFT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcSizeOfExpr xmobj)
    {
        return enterSizeOrAlignExpr(Xcode.SIZE_OF_EXPR, xmobj, xmobj.getType());
    }

    @Override
    public boolean enter(XbcStatementLabel xmobj)
    {
        return enterOneArgObj(Xcode.STATEMENT_LABEL, xmobj,
            new SymbolObj(xmobj.getName()));
    }

    @Override
    public boolean enter(XbcStringConstant xmobj)
    {
        _xobj = Xcons.StringConstant(xmobj.getContent());
        return true;
    }
   
    @Override
    public boolean enter(XbcStructType xmobj)
    {
        XobjList identList = (XobjList)toXobject(xmobj.getSymbols());
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        Xtype type = new StructType(xmobj.getType(), identList,
            getTypeQualFlags(xmobj), gccAttrs);
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcSubArrayRef xmobj)
    {
        _xobj = Xcons.List(Xcode.SUB_ARRAY_REF, getType(xmobj),
                           toXobject(xmobj.getArrayAddr()),
			   xmobj.getSubArrayDimension());
//         _xobj = Xcons.List(Xcode.SUB_ARRAY_REF, getType(xmobj),
//                            toXobject(xmobj.getArrayAddr()),
//                           toXobjList(Xcode.LIST, getType(xmobj), xmobj.getSubArrayDimension()));
        return true;
    }

    @Override
    public boolean enter(XbcIndexRange xmobj) {
      return enterAsXobjList(xmobj, Xcode.LIST /*Xcode.INDEX_RANGE*/,
                               xmobj.getLowerBound(),
                               xmobj.getUpperBound(),
                               xmobj.getStep());
    }

    @Override
    public boolean enter(XbcAddrOfExpr xmobj) {
        return enterAsXobjList(xmobj, Xcode.ADDR_OF, xmobj.getExpressions());
    }

//     @Override
//     public boolean enter(XbcLowerBound xmobj)
//     {
//         return enterAsXobjList(xmobj, Xcode.LOWER_BOUND, xmobj.getExpressions());
//     }

//     @Override
//     public boolean enter(XbcStep xmobj)
//     {
//         return enterAsXobjList(xmobj, Xcode.STEP, xmobj.getExpressions());
//     }

//     @Override
//     public boolean enter(XbcUpperBound xmobj)
//     {
//         return enterAsXobjList(xmobj, Xcode.UPPER_BOUND, xmobj.getExpressions());
//     }

    @Override
    public boolean enter(XbcSwitchStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.SWITCH_STATEMENT,
            xmobj.getValue(),
            xmobj.getBody());
    }

    @Override
    public boolean enter(XbcSymbols xmobj)
    {
        Xobject identList = new XobjList(Xcode.ID_LIST);
        
        for(IXbcSymbolsChoice choice : xmobj.getContent()) {
            if(choice instanceof XbcId) {
                identList.add(toXobject(choice));
            } else if((choice instanceof XbcPragma) || (choice instanceof XbcText)){
                warning((XmObj)choice, "ignored. directive is not allowed here");
            }
        }
        
        _xobj = identList;
        
        return true;
    }

    @Override
    public boolean enter(XbcText xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.TEXT,
            new StringObj(Xcode.STRING, xmobj.getContent()));
    }

    @Override
    public boolean enter(XbcThen xmobj)
    {
        return enterChild(xmobj.getStatements());
    }

    @Override
    public boolean enter(XbcTypeName xmobj)
    {
        _xobj = toXobjList(Xcode.TYPE_NAME, getType(xmobj.getType()), new IRVisitable[0]);
        return true;
    }

    @Override
    public boolean enter(XbcTypeTable xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST, xmobj.getTypes());
    }

    @Override
    public boolean enter(XbcUnaryMinusExpr xmobj)
    {
        return enterUnaryExpr(Xcode.UNARY_MINUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbcUnionType xmobj)
    {
        XobjList identList = (XobjList)toXobject(xmobj.getSymbols());
        Xobject gccAttrs = toXobject(xmobj.getGccAttributes());
        Xtype type = new UnionType(xmobj.getType(), identList,
            getTypeQualFlags(xmobj), gccAttrs);
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbcValue xmobj)
    {
        // XbcValue is top level 'value' element.
        return enterAsXobject(xmobj.getContent());
    }

    @Override
    public boolean enter(XbcVar xmobj)
    {
        return enterVar(Xcode.VAR, xmobj);
    }

    @Override
    public boolean enter(XbcVarAddr xmobj)
    {
        return enterSymbolAddr(Xcode.VAR_ADDR, xmobj, xmobj.getScope());
    }

    @Override
    public boolean enter(XbcVarDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.VAR_DECL,
            new SymbolObj(xmobj.getName()),
            xmobj.getValue(),
            xmobj.getGccAsm());
    }

    @Override
    public boolean enter(XbcWhileStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.WHILE_STATEMENT,
            xmobj.getCondition(),
            xmobj.getBody());
    }

    @Override
    public boolean enter(XbcXcodeProgram xmobj)
    {
        XobjectFile xobjFile = new XobjectFile();
        _xobjFile = xobjFile;
        
        // program attributes
        xobjFile.setProgramAttributes(
            xmobj.getSource(), xmobj.getLanguage(),
            xmobj.getCompilerInfo(), xmobj.getVersion(), xmobj.getTime());

        // type table
        _typeTable = xmobj.getTypeTable();
        toXobject(_typeTable);
        xobjFile.fixupTypeRef();
        
        // symbol table
        Xobject identList = toXobject(xmobj.getGlobalSymbols());
        if(identList != null)
            xobjFile.setIdentList(identList);

        // global declarations
        Xobject declList = toXobject(xmobj.getGlobalDeclarations());
        if(declList != null)
            xobjFile.add(declList);

        return true;
    }
}
