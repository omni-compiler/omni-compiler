/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import static xcodeml.util.XmStringUtil.trim;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import exc.object.*;
import exc.util.XobjectRecursiveVisitor;
import exc.util.XobjectVisitor;
import xcodeml.IXobject;
import xcodeml.XmException;
import xcodeml.XmObj;
import xcodeml.util.XmBindingException;
import xcodeml.util.XmLog;
import xcodeml.util.XmXmObjToXobjectTranslator;

import xcodeml.binding.IRNode;
import xcodeml.binding.IXbLineNo;
import xcodeml.binding.IXbTypedExpr;
import xcodeml.f.binding.*;
import xcodeml.f.binding.gen.*;

/**
 * Visitor for XcodeML/Fortran to Xcode translation.
 */
public class XmfXmObjToXobjectTranslator extends RVisitorBase
    implements XmXmObjToXobjectTranslator
{
    private Xobject _xobj;
    
    private XobjectFile _xobjFile;
    
    private XbfTypeTable _typeTable;

    private static final boolean debug_pre_transform = false;
    
    public XmfXmObjToXobjectTranslator()
    {
    }
    
    public XmfXmObjToXobjectTranslator(XobjectFile xobjFile, XbfTypeTable typeTable)
    {
        _xobjFile = xobjFile;
        _typeTable = typeTable;
    }

    @Override
    public IXobject translate(XmObj xmobj) throws XmException
    {
        try {
            ((IRVisitable)xmobj).enter(this);
            if(!parsePragma())
                return null;
            getIXobject().setParentRecursively(null);
            if(getIXobject() instanceof XobjectFile) {
                setIdentDecl((XobjectFile)getIXobject());
            }
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
    
    private void setIdentDecl(XobjectFile objFile)
    {
        topdownXobjectDefIterator ite = new topdownXobjectDefIterator(objFile);
        for(ite.init(); !ite.end(); ite.next()) {
            XobjectDef def = ite.getDef();
            if(!def.isFuncDef() && !def.isFmoduleDef())
                continue;
            Xobject decls = def.getFuncDecls();
            if(decls == null)
                continue;
            for(Xobject decl : (XobjList)decls) {
                if(decl.Opcode() == Xcode.VAR_DECL) {
                    String name = decl.getArg(0).getName();
                    Ident id = def.findIdent(name, IXobject.FINDKIND_VAR);
                    if(id != null && id.Type() != null &&
                        id.Type().isFparameter()) {
                        id.setFparamValue(decl.getArgOrNull(1));
                    }
                }
            }
        }
    }

    private boolean parsePragma()
    {
        if(_xobjFile == null)
            return true;
        
        XobjectVisitor xvisitor = new ParsePragmaVisitor();
        
        if(debug_pre_transform) {
            Writer writer = new java.io.OutputStreamWriter(System.out);
            _xobjFile.Output(writer);
            try { writer.flush(); } catch(Exception e) {}
        }
        
        for(XobjectDef def : _xobjFile) {
            if(!def.enter(xvisitor))
                return false;
        }
        
        return true;
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
        
        private XobjArgs findPostfix(XobjArgs prefixArgs)
        {
            Xobject prefix = prefixArgs.getArg();
            Xcode prefixOp = prefix.Opcode();
            int stackCount = 0;
            for(XobjArgs a = prefixArgs.nextArgs(); a != null; a = a.nextArgs()) {
                Xobject x = a.getArg();
                if(x == null || prefixOp != x.Opcode())
                    continue;
                PragmaSyntax syn = PragmaSyntax.valueOrNullOf(x.getArg(0).getString());
                if(parser.isPrePostPair(prefix, x)) {
                    if(syn == PragmaSyntax.SYN_POSTFIX) {
                        if(stackCount == 0)
                            return a;
                        else
                            stackCount--;
                    } else if(syn == PragmaSyntax.SYN_START) {
                        stackCount++;
                    }
                }
            }
            
            return parser.getAbbrevPostfix(prefixArgs);
        }
        
        private void collectCommonDeclIdent(Xobject cmnDecl)
        {
            for(Xobject varList : (XobjList)cmnDecl) {
                Xobject cmnName = varList.getArg(0);
                if(cmnName == null)
                    continue;
                Ident cmnId = parser.findIdent(cmnName.getName(), IXobject.FINDKIND_COMMON);
                if(cmnId == null)
                    continue;
                if(cmnId.getStorageClass() != StorageClass.FCOMMON_NAME)
                    throw new IllegalStateException("illegal storage class : " + cmnId.getName() + " : "
                        + cmnId.getStorageClass().toString());
                for(Xobject varRef : (XobjList)varList.getArg(1)) {
                    Xobject var = varRef.getArg(0);
                    Ident varId = parser.findIdent(var.getName(), IXobject.FINDKIND_VAR);
                    if(varId == null)
                        continue;
                    if(varId.getStorageClass() != StorageClass.FCOMMON)
                        throw new IllegalStateException("illegal storage class : " + varId.getName() + " : "
                            + varId.getStorageClass().toString());
                    cmnId.addFcommonVar(varId);
                    varId.setFcommonName(cmnId.getName());
                }
            }
        }
        
        @Override
        public boolean enter(Xobject vv)
        {
            if(!(vv instanceof XobjList))
                return true;

            XobjList v = (XobjList)vv;
            XobjList identList = null;
            
            switch(v.Opcode()) {
            case FUNCTION_DEFINITION:
            case F_MODULE_DEFINITION:
                identList = v.getIdentList();
                break;
            case F_COMMON_DECL:
                collectCommonDeclIdent(v);
                return true;
            }
            
            if(identList != null)
                parser.pushEnv(identList);

            for(XobjArgs a = v.getArgs(); a != null; a = a.nextArgs()) {
                Xobject x = a.getArg();
                
                if(x == null || !x.isPragma() || x.Opcode() == Xcode.PRAGMA_LINE ||
                    x.isParsed())
                    continue;

                PragmaSyntax syn = PragmaSyntax.valueOrNullOf(x.getArg(0).getString());
                if(syn == null)
                    continue;
                x.removeFirstArgs();
                
                if(syn == PragmaSyntax.SYN_START) {
                    XobjArgs postfixArg = findPostfix(a);
                    if(postfixArg == null) {
                        XmLog.error(x.getLineNo(), "pragma end clause is not found.");
                        return false;
                    }
                    
                    parser.mergeStartAndPostfixArgs(x, postfixArg.getArg());
                    
                    Xobject body = Xcons.List(Xcode.F_STATEMENT_LIST);
                    for(XobjArgs aa = a.nextArgs(); aa != null && aa != postfixArg; aa = aa.nextArgs()) {
                        body.add(aa.getArg());
                    }
                    x.add(body);
                    a.setNext(postfixArg.nextArgs());
                    postfixArg.setArg(null);
                    
                    parser.completePragmaEnd(x, body);
                } else if(syn == PragmaSyntax.SYN_PREFIX) {
                    XobjArgs nextArg = a.nextArgs();
                    if(nextArg == null) {
                        XmLog.error(x.getLineNo(), "no statement next to pragma");
                        return false;
                    }
                    x.add(nextArg.getArg());
                    a.setNext(nextArg.nextArgs());
                } else if(syn == PragmaSyntax.SYN_POSTFIX) {
                    XmLog.error(x.getLineNo(), "unexpected pragma end clause");
                    return false;
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
            
            if(identList != null)
                parser.popEnv();

            return true;
        }
    }

    class CompileException extends RuntimeException
    {
        private static final long serialVersionUID = -6715840674962886607L;
        
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
            if(xmobj != null && xmobj.getType() != null)
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
        
        SymbolObj(XbfName name)
        {
            this(Xcode.IDENT, name, name.getContent());
        }
        
        SymbolObj(Xcode code, String name)
        {
            this(code, null, name, null);
        }
        
        SymbolObj(String name)
        {
            this((IXbTypedExpr)null, name);
        }
        
        @Override
        public boolean enter(IRVisitor visitor)
        {
            if(_name != null) {
                ((XmfXmObjToXobjectTranslator)visitor)._xobj =
                    Xcons.Symbol(_code, _type, _name, _scope);
            }
            return true;
        }
    }
    
    private class StringObj extends DefaultObj
    {
        private Xcode _code;
        private String _value;
        private Xtype _type;
        
        StringObj(Xcode code, Xtype type, String value)
        {
            _code = code;
            _type = type;
            _value = value;
        }
        
        StringObj(Xcode code, String value)
        {
            this(code, null, value);
        }
        
        StringObj(String value)
        {
            this(Xcode.F_CHARACTER_CONSTATNT, Xtype.FcharacterType, value);
        }
        
        @Override
        public boolean enter(IRVisitor visitor)
        {
            if(_value != null) {
                ((XmfXmObjToXobjectTranslator)visitor)._xobj =
                    new XobjString(_code, _type, _value);
            }
            return true;
        }
    }

    private class IntFlagObj extends DefaultObj
    {
        private boolean _value;
        
        IntFlagObj(boolean value)
        {
            _value = value;
        }

        @Override
        public boolean enter(IRVisitor visitor)
        {
            ((XmfXmObjToXobjectTranslator)visitor)._xobj =
                Xcons.Int(Xcode.INT_CONSTANT, (_value ? 1 : 0));
            return true;
        }
    }
    
    private class ListObj extends DefaultObj
    {
        private IRVisitable[] _elems;
                    
        ListObj(IRVisitable[] elems)
        {
            _elems = elems;
        }
        
        @Override
        public boolean enter(IRVisitor visitor)
        {
            ((XmfXmObjToXobjectTranslator)visitor)._xobj =
                ((XmfXmObjToXobjectTranslator)visitor).toXobjList(Xcode.LIST, null, _elems);
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
    
    private boolean enterChild(IRVisitable xmobj, IRVisitable childXmobj)
    {
        if(xmobj != null)
            setCommonAttributes(xmobj);
        if(childXmobj != null)
            childXmobj.enter(this);
        return true;
    }
    
    private Xobject toXobject(IRVisitable xmobj)
    {
        if(xmobj == null)
            return null;
        
        XmfXmObjToXobjectTranslator t = new XmfXmObjToXobjectTranslator(_xobjFile, _typeTable);
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

    private boolean enterAsXobjList(IRVisitable xmobj, Xcode code, IRVisitable ... childXmobjs)
    {
        Xtype type = null;
        if(xmobj instanceof IXbTypedExpr) {
            String tid = ((IXbTypedExpr)xmobj).getType();
            if(tid != null)
                type = getType(tid);
        }
        if(childXmobjs != null)
            _xobj = toXobjList(code, type, childXmobjs);
        else
            _xobj = toXobjList(code, type);
        setCommonAttributes(xmobj);
        return true;
    }
    
    private void setCommonAttributes(IRVisitable xmobj)
    {
        if(xmobj instanceof IXbLineNo) {
            IXbLineNo i = (IXbLineNo)xmobj;
            if(i.getLineno() != null) {
                _xobj.setLineNo(new LineNo(
                    i.getFile(), Integer.parseInt(i.getLineno())));
            }
        }
    }
    
    private boolean enterUnaryExpr(Xcode code, IXbfUnaryExpr xmobj)
    {
        return enterAsXobjList((IRVisitable)xmobj, code, xmobj.getDefModelExpr());
    }
    
    private boolean enterBinaryExpr(Xcode code, IXbfBinaryExpr xmobj)
    {
        return enterAsXobjList((IRVisitable)xmobj, code,
            xmobj.getDefModelExpr1(), xmobj.getDefModelExpr2());
    }
    
    private boolean enterIOStatement(Xcode code, IXbfIOStatement xmobj)
    {
        return enterAsXobjList((IRVisitable)xmobj, code,
            xmobj.getNamedValueList());
    }

    private boolean enterRWStatement(Xcode code, IXbfRWStatement xmobj)
    {
        return enterAsXobjList((IRVisitable)xmobj, code,
            xmobj.getNamedValueList(),
            xmobj.getValueList());
    }
    
    /* utility methods end
     * -------------------------------------------------
     */

    @Override
    public boolean enter(XbfXcodeProgram xmobj)
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
    
    @Override
    public boolean enter(XbfTypeTable xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST, xmobj.getContent());
    }

    @Override
    public boolean enter(XbfFbasicType xmobj)
    {
        String tid = xmobj.getType();
        BasicType.TypeInfo ti = BasicType.getTypeInfoByFName(xmobj.getRef());
        int tq =
            (xmobj.getIsAllocatable() ? Xtype.TQ_FALLOCATABLE : 0) |
            (xmobj.getIsOptional() ? Xtype.TQ_FOPTIONAL : 0) |
            (xmobj.getIsParameter() ? Xtype.TQ_FPARAMETER : 0) |
            (xmobj.getIsPointer() ? Xtype.TQ_FPOINTER : 0) |
            (xmobj.getIsPrivate() ? Xtype.TQ_FPRIVATE : 0) |
            (xmobj.getIsPublic() ? Xtype.TQ_FPUBLIC : 0) |
            (xmobj.getIsSave() ? Xtype.TQ_FSAVE : 0) |
            (xmobj.getIsTarget() ? Xtype.TQ_FTARGET : 0) |
            (xmobj.getIsCrayPointer() ? Xtype.TQ_FCRAY_POINTER : 0);

        String intent = xmobj.getIntent();
        if(intent != null) {
            if(XbfFbasicType.INTENT_IN.equals(intent)) {
                tq |= Xtype.TQ_FINTENT_IN;
            }
            else if(XbfFbasicType.INTENT_OUT.equals(intent)) {
                tq |= Xtype.TQ_FINTENT_OUT;
            }
            else if(XbfFbasicType.INTENT_INOUT.equals(intent)) {
                tq |= Xtype.TQ_FINTENT_INOUT;
            }
        }
        
        Xobject fkind = toXobject(xmobj.getKind());
        IXbfFbasicTypeChoice choice = xmobj.getContent();
        Xobject flen = null, sizeExprs[] = null;
        
        if(choice instanceof XbfLen) {
            flen = toXobject(xmobj.getContent());
            if(flen == null)
                flen = Xcons.IntConstant(-1); // means variable length
        } else if(choice instanceof XbfDefModelArraySubscriptSequence1) {
            List<Xobject> sizeExprList = new ArrayList<Xobject>();
            for(IXbfDefModelArraySubscriptChoice c :
                ((XbfDefModelArraySubscriptSequence1)choice).getDefModelArraySubscript()) {
                sizeExprList.add(toXobject(c));
            }
            sizeExprs = sizeExprList.toArray(new Xobject[0]);
        } else if(choice != null) {
            XmLog.fatal("FbasicType children : " + choice);
            return false;
        }
        
        Xtype type;
        
        if(sizeExprs == null) {
            if(ti == null) {
                // inherited type
                Xtype ref = getType(xmobj.getRef());
                type = ref.inherit(tid);
                type.setTypeQualFlags(tq);
            } else {
                type = new BasicType(ti.type.getBasicType(), tid, tq, null, fkind, flen);
            }
        } else {
            
            Xtype ref = getType(xmobj.getRef());
            type = new FarrayType(tid, ref, tq, sizeExprs);
        }
        
        getXobjectFile().addType(type);
        
        return true;
    }
    
    @Override
    public boolean enter(XbfDefModelArraySubscriptSequence1 xmobj)
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean enter(XbfFfunctionType xmobj)
    {
        String tid = xmobj.getType();
        Xtype retType = getType(xmobj.getReturnType());
        int tq =
            (xmobj.getIsExternal() ? Xtype.TQ_FEXTERNAL : 0) |
            (xmobj.getIsInternal() ? Xtype.TQ_FINTERNAL : 0) |
            (xmobj.getIsIntrinsic() ? Xtype.TQ_FINTRINSIC : 0) |
            (xmobj.getIsPrivate() ? Xtype.TQ_FPRIVATE : 0) |
            (xmobj.getIsPublic() ? Xtype.TQ_FPUBLIC : 0) |
            (xmobj.getIsProgram() ? Xtype.TQ_FPROGRAM : 0) |
            (xmobj.getIsRecursive() ? Xtype.TQ_FRECURSIVE : 0);
        
        Xobject params = toXobject(xmobj.getParams());
        FunctionType type = new FunctionType(tid, retType,
            params, tq, false, null, xmobj.getResultName());
        
        getXobjectFile().addType(type);

        return true;
    }

    @Override
    public boolean enter(XbfParams xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST, xmobj.getName());
    }

    @Override
    public boolean enter(XbfFstructType xmobj)
    {
        String tid = xmobj.getType();
        int tq =
            (xmobj.getIsInternalPrivate() ? Xtype.TQ_FINTERNAL_PRIVATE : 0) |
            (xmobj.getIsPrivate() ? Xtype.TQ_FPRIVATE : 0) |
            (xmobj.getIsPublic() ? Xtype.TQ_FPUBLIC : 0) |
            (xmobj.getIsSequence() ? Xtype.TQ_FSEQUENCE : 0);
        
        XobjList id_list = (XobjList)toXobject(xmobj.getSymbols());
        StructType type = new StructType(tid, id_list, tq, null);
        getXobjectFile().addType(type);
        
        return true;
    }

    @Override
    public boolean enter(XbfGlobalDeclarations xmobj)
    {
        Xobject decls = toXobjList(Xcode.LIST, null, xmobj.getContent());
        if(decls != null)
            getXobjectFile().add(decls);
        return true;
    }

    @Override
    public boolean enter(XbfDeclarations xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST, xmobj.getContent());
    }

    @Override
    public boolean enter(XbfGlobalSymbols xmobj)
    {
        Xobject identList = new XobjList(Xcode.ID_LIST);
        
        for(XbfId xmId : xmobj.getId()) {
            identList.add(toXobject(xmId));
        }
        
        _xobj = identList;
        
        return true;
    }

    @Override
    public boolean enter(XbfSymbols xmobj)
    {
        Xobject identList = new XobjList(Xcode.ID_LIST);
        
        for(XbfId id : xmobj.getId()) {
            identList.add(toXobject(id));
        }
        
        _xobj = identList;
        
        return true;
    }

    @Override
    public boolean enter(XbfId xmobj)
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
        Xobject addr = null;
        
        if(sclassStr != null) {
            sclass = StorageClass.get(sclassStr);

            switch(sclass) {
            case FLOCAL:
            case FSAVE:
            case FCOMMON:
            case FPARAM:
                addr = Xcons.Symbol(Xcode.VAR, type, name);
                addr.setScope(sclass == StorageClass.FPARAM ?
                    VarScope.PARAM : VarScope.LOCAL);
                break;
            case FFUNC:
                addr = Xcons.Symbol(Xcode.FUNC_ADDR, type, name);
                addr.setScope(VarScope.LOCAL);
                break;
            }
        }
        
        // create ident
        Ident ident = new Ident(name, sclass, type, addr,
            0, null, 0, null, null, null);
        
        _xobj = ident;
        
        if(type != null && StorageClass.FTYPE_NAME.equals(sclass))
        	type.setTagIdent(ident);
        
        // declaring
        if(sclass != null && sclass.isVarOrFunc()) {
            ident.setIsDeclared(true);
        }
        
        return true;
    }

    @Override
    public boolean enter(XbfName xmobj)
    {
        return new SymbolObj(xmobj).enter(this);
    }

    @Override
    public boolean enter(XbfKind xmobj)
    {
        return enterChild(xmobj, xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfLen xmobj)
    {
        return enterChild(xmobj, xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfArrayIndex xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_ARRAY_INDEX,
            xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfIndexRange xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_INDEX_RANGE,
            xmobj.getLowerBound(), xmobj.getUpperBound(), xmobj.getStep(),
            new IntFlagObj(xmobj.getIsAssumedShape()),
            new IntFlagObj(xmobj.getIsAssumedSize()));
    }

    @Override
    public boolean enter(XbfLowerBound xmobj)
    {
        return enterChild(xmobj, xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfUpperBound xmobj)
    {
        return enterChild(xmobj, xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfStep xmobj)
    {
        return enterChild(xmobj, xmobj.getDefModelExpr());
    }
    
    private void markModuleVariable(XobjList ids, XobjList decls)
    {
        if(ids == null)
            return;
        // mark module variable
        Set<String> declSet = new HashSet<String>();
        if(decls != null) {
            for(Xobject a : decls) {
                if(a.Opcode() == Xcode.VAR_DECL)
                    declSet.add(a.getArg(0).getName());
            }
        }
        
        for(Xobject a : ids) {
            Ident id = (Ident)a;
            if((id.getStorageClass() == StorageClass.FSAVE ||
                id.getStorageClass() == StorageClass.FLOCAL) &&
                !declSet.contains(id.getName())) {
                id.setIsFmoduleVar(true);
            }
        }
    }

    @Override
    public boolean enter(XbfFfunctionDefinition xmobj)
    {
        boolean r = enterAsXobjList(xmobj, Xcode.FUNCTION_DEFINITION,
            new SymbolObj(xmobj.getName()),
            xmobj.getSymbols(),
            xmobj.getDeclarations(),
            xmobj.getBody(),
            null);

        markModuleVariable((XobjList)_xobj.getArgOrNull(1),
            (XobjList)_xobj.getArgOrNull(2));
        
        return r;
    }

    @Override
    public boolean enter(XbfVarDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.VAR_DECL,
            new SymbolObj(xmobj.getName()),
            xmobj.getValue(), null);
    }

    @Override
    public boolean enter(XbfFfunctionDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.FUNCTION_DECL,
            new SymbolObj(xmobj.getName()), null, null,
            xmobj.getDeclarations());
    }

    @Override
    public boolean enter(XbfFdataDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_DATA_DECL,
            xmobj.getFdataDeclSequence());
            
    }

    @Override
    public boolean enter(XbfFdataDeclSequence xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST,
            xmobj.getVarList(), xmobj.getValueList());
    }

    @Override
    public boolean enter(XbfVarList xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_VAR_LIST,
            new SymbolObj(xmobj.getName()),
            new ListObj(xmobj.getContent()));
    }

    @Override
    public boolean enter(XbfValueList xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_VALUE_LIST,
            xmobj.getValue());
    }

    @Override
    public boolean enter(XbfValue xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_VALUE,
            xmobj.getDefModelExpr(), xmobj.getRepeatCount());
    }
    
    @Override
    public boolean enter(XbfRepeatCount xmobj)
    {
        return enterChild(xmobj, xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfFdoLoop xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_DO_LOOP,
            xmobj.getVar(), xmobj.getIndexRange(),
            new ListObj(xmobj.getValue()));
    }

    @Override
    public boolean enter(XbfFblockDataDefinition xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_BLOCK_DATA_DEFINITION,
            new SymbolObj(xmobj.getName()), xmobj.getSymbols(),
            xmobj.getDeclarations());
    }
    
    @Override
    public boolean enter(XbfFentryDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_ENTRY_DECL,
            new SymbolObj(xmobj.getName()));
    }

    @Override
    public boolean enter(XbfExternDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_EXTERN_DECL,
            new SymbolObj(xmobj.getName()));
    }

    @Override
    public boolean enter(XbfFequivalenceDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_EQUIVALENCE_DECL,
            xmobj.getFequivalenceDeclSequence());
    }

    @Override
    public boolean enter(XbfFequivalenceDeclSequence xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST,
            xmobj.getVarRef(), xmobj.getVarList());
    }

    @Override
    public boolean enter(XbfFcommonDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_COMMON_DECL,
            xmobj.getVarList());
    }

    @Override
    public boolean enter(XbfFmoduleDefinition xmobj)
    {
        boolean r = enterAsXobjList(xmobj, Xcode.F_MODULE_DEFINITION,
            new SymbolObj(xmobj.getName()), xmobj.getSymbols(),
            xmobj.getDeclarations(), xmobj.getFcontainsStatement());
        
        markModuleVariable((XobjList)_xobj.getArgOrNull(1),
            (XobjList)_xobj.getArgOrNull(2));
        
        return r;
    }

    @Override
    public boolean enter(XbfFmoduleProcedureDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_MODULE_PROCEDURE_DECL,
            xmobj.getName());
    }

    @Override
    public boolean enter(XbfFinterfaceDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_INTERFACE_DECL,
            new SymbolObj(xmobj.getName()),
            new IntFlagObj(xmobj.getIsOperator()),
            new IntFlagObj(xmobj.getIsAssignment()),
            new ListObj(xmobj.getContent()));
    }

    @Override
    public boolean enter(XbfFformatDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_FORMAT_DECL,
            new StringObj(Xcode.STRING, xmobj.getFormat()));
    }

    @Override
    public boolean enter(XbfFnamelistDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_NAMELIST_DECL,
            xmobj.getVarList());
    }

    @Override
    public boolean enter(XbfFstructDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_STRUCT_DECL,
            new SymbolObj(xmobj.getName()));
    }

    @Override
    public boolean enter(XbfFuseDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_USE_DECL,
            new SymbolObj(xmobj.getName()),
            new ListObj(xmobj.getRename()));
    }

    @Override
    public boolean enter(XbfFuseOnlyDecl xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_USE_ONLY_DECL,
            new SymbolObj(xmobj.getName()),
            new ListObj(xmobj.getRenamable()));
    }

    @Override
    public boolean enter(XbfRename xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_RENAME,
            new SymbolObj(xmobj.getUseName()),
            new SymbolObj(xmobj.getLocalName()));
    }

    @Override
    public boolean enter(XbfRenamable xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_RENAMABLE,
            new SymbolObj(xmobj.getUseName()),
            new SymbolObj(xmobj.getLocalName()));
    }

    @Override
    public boolean enter(XbfExprStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.EXPR_STATEMENT,
            xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfFassignStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_ASSIGN_STATEMENT,
            xmobj.getDefModelLValue(), xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfFpointerAssignStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_POINTER_ASSIGN_STATEMENT,
            xmobj.getDefModelLValue(), xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfFdoStatement xmobj)
    {
        IRVisitable xmvar = null, xmir = null;
        
        if(xmobj.getFdoStatementSequence() != null) {
            xmvar = xmobj.getFdoStatementSequence().getVar();
            xmir = xmobj.getFdoStatementSequence().getIndexRange();
        }
        return enterAsXobjList(xmobj, Xcode.F_DO_STATEMENT,
            new SymbolObj(xmobj.getConstructName()),
            xmvar, xmir, xmobj.getBody());
    }

    @Override
    public boolean enter(XbfFdoStatementSequence xmobj)
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean enter(XbfFdoWhileStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_DO_WHILE_STATEMENT,
            new SymbolObj(xmobj.getConstructName()),
            xmobj.getCondition(), xmobj.getBody());
    }

    @Override
    public boolean enter(XbfCondition xmobj)
    {
        return enterChild(xmobj, xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfFselectCaseStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_SELECT_CASE_STATEMENT,
            new SymbolObj(xmobj.getConstructName()),
            xmobj.getValue(),
            new ListObj(xmobj.getFcaseLabel()));
    }

    @Override
    public boolean enter(XbfFcaseLabel xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_CASE_LABEL,
            new SymbolObj(xmobj.getConstructName()),
            new ListObj(xmobj.getContent()),
            xmobj.getBody());
    }

    @Override
    public boolean enter(XbfFwhereStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_WHERE_STATEMENT,
            null, xmobj.getCondition(), xmobj.getThen(), xmobj.getElse());
    }

    @Override
    public boolean enter(XbfThen xmobj)
    {
        return enterChild(xmobj, xmobj.getBody());
    }

    @Override
    public boolean enter(XbfElse xmobj)
    {
        return enterChild(xmobj, xmobj.getBody());
    }

    @Override
    public boolean enter(XbfFifStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_IF_STATEMENT,
            new SymbolObj(xmobj.getConstructName()),
            xmobj.getCondition(), xmobj.getThen(), xmobj.getElse());
    }

    @Override
    public boolean enter(XbfBody xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_STATEMENT_LIST,
            xmobj.getDefModelStatement());
    }

    @Override
    public boolean enter(XbfFcycleStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_CYCLE_STATEMENT,
            new SymbolObj(xmobj.getConstructName()));
    }

    @Override
    public boolean enter(XbfFexitStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_EXIT_STATEMENT,
            new SymbolObj(xmobj.getConstructName()));
    }

    @Override
    public boolean enter(XbfContinueStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_CONTINUE_STATEMENT);
    }

    @Override
    public boolean enter(XbfFreturnStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.RETURN_STATEMENT);
    }

    @Override
    public boolean enter(XbfGotoStatement xmobj)
    {
        IRVisitable xmvalue = null, xmparams = null;
        
        if(xmobj.getGotoStatementSequence() != null) {
            xmvalue = xmobj.getGotoStatementSequence().getValue();
            xmparams = xmobj.getGotoStatementSequence().getParams();
        }
        
        return enterAsXobjList(xmobj, Xcode.GOTO_STATEMENT,
            new SymbolObj(xmobj.getLabelName()), xmvalue, xmparams);
    }

    @Override
    public boolean enter(XbfGotoStatementSequence xmobj)
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean enter(XbfStatementLabel xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.STATEMENT_LABEL,
            new SymbolObj(xmobj.getLabelName()));
    }

    @Override
    public boolean enter(XbfFcontainsStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_CONTAINS_STATEMENT,
            xmobj.getFfunctionDefinition());
    }

    @Override
    public boolean enter(XbfFpragmaStatement xmobj)
    {
        PragmaLexer lexer = new FpragmaLexer(this, xmobj);
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
    public boolean enter(XbfFallocateStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_ALLOCATE_STATEMENT,
            new SymbolObj(xmobj.getStatName()),
            new ListObj(xmobj.getAlloc()));
    }

    @Override
    public boolean enter(XbfFdeallocateStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_DEALLOCATE_STATEMENT,
            new SymbolObj(xmobj.getStatName()),
            new ListObj(xmobj.getAlloc()));
    }

    @Override
    public boolean enter(XbfFnullifyStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_NULLIFY_STATEMENT,
            xmobj.getAlloc());
    }
  
    @Override
    public boolean enter(XbfAlloc xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_ALLOC,
            xmobj.getContent(),
            new ListObj(xmobj.getDefModelArraySubscript()));
    }

    @Override
    public boolean enter(XbfFbackspaceStatement xmobj)
    {
        return enterIOStatement(Xcode.F_BACKSPACE_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfNamedValueList xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_NAMED_VALUE_LIST,
            xmobj.getNamedValue());
    }
    
    @Override
    public boolean enter(XbfNamedValue xmobj)
    {
        IRVisitable xmval;
        String val1 = xmobj.getValue();
        
        if(val1 != null) {
            xmval = new StringObj(Xcode.STRING, val1);
        } else {
            xmval = xmobj.getDefModelExpr();
        }
        return enterAsXobjList(xmobj, Xcode.F_NAMED_VALUE,
            new SymbolObj(xmobj.getName()), xmval);
    }

    @Override
    public boolean enter(XbfFcloseStatement xmobj)
    {
        return enterIOStatement(Xcode.F_CLOSE_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfFendFileStatement xmobj)
    {
        return enterIOStatement(Xcode.F_END_FILE_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfFinquireStatement xmobj)
    {
        return enterRWStatement(Xcode.F_INQUIRE_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfFopenStatement xmobj)
    {
        return enterIOStatement(Xcode.F_OPEN_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfFprintStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_PRINT_STATEMENT,
            new StringObj(Xcode.STRING, xmobj.getFormat()),
            xmobj.getValueList());
    }

    @Override
    public boolean enter(XbfFrewindStatement xmobj)
    {
        return enterIOStatement(Xcode.F_REWIND_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfFreadStatement xmobj)
    {
        return enterRWStatement(Xcode.F_READ_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfFwriteStatement xmobj)
    {
        return enterRWStatement(Xcode.F_WRITE_STATEMENT, xmobj);
    }

    @Override
    public boolean enter(XbfFpauseStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_PAUSE_STATEMENT,
            new StringObj(Xcode.STRING, xmobj.getCode()),
            new StringObj(xmobj.getMessage()));
    }

    @Override
    public boolean enter(XbfFstopStatement xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_STOP_STATEMENT,
            new StringObj(Xcode.STRING, xmobj.getCode()),
            new StringObj(xmobj.getMessage()));
    }

    @Override
    public boolean enter(XbfFcharacterRef xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_CHARACTER_REF,
            xmobj.getVarRef(), xmobj.getIndexRange());
    }

    @Override
    public boolean enter(XbfFconcatExpr xmobj)
    {
        return enterBinaryExpr(Xcode.F_CONCAT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfFintConstant xmobj)
    {
        BigInteger bi = new BigInteger(xmobj.getContent());
        int bl = bi.bitLength();
        Xtype type = getType(xmobj.getType());
        String kind = xmobj.getKind();
        
        if(bl <= 31) {
            _xobj = Xcons.IntConstant(bi.intValue(), type, kind);
        } else {
            _xobj = Xcons.LongLongConstant(bi, type, kind);
        }
        
        return true;
    }

    @Override
    public boolean enter(XbfFrealConstant xmobj)
    {
        Xtype type = getType(xmobj.getType());
        try {
            _xobj = Xcons.FloatConstant(type, xmobj.getContent(), xmobj.getKind());
        } catch(Exception e) {
            throw new XmBindingException(xmobj, e);
        }
        
        return true;
    }

    @Override
    public boolean enter(XbfFcharacterConstant xmobj)
    {
        Xtype type = getType(xmobj.getType());
        _xobj = Xcons.FcharacterConstant(type, xmobj.getContent(), xmobj.getKind());
        
        return true;
    }

    @Override
    public boolean enter(XbfFlogicalConstant xmobj)
    {
        Xtype type = getType(xmobj.getType());
        String valueStr = xmobj.getContent();
        boolean value = false;
        if(valueStr.equalsIgnoreCase(".TRUE."))
            value = true;
        _xobj = Xcons.FlogicalConstant(type, value, xmobj.getKind());
        
        return true;
    }

    @Override
    public boolean enter(XbfFcomplexConstant xmobj)
    {
        Xtype type = getType(xmobj.getType());
        _xobj = Xcons.ComplexConstant(type,
            toXobject(xmobj.getDefModelExpr1()), toXobject(xmobj.getDefModelExpr2()));
        return true;
    }

    @Override
    public boolean enter(XbfFfunction xmobj)
    {
        return enterChild(xmobj, new SymbolObj(Xcode.FUNC_ADDR, xmobj.getContent()));
    }

    @Override
    public boolean enter(XbfFarrayRef xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_ARRAY_REF,
            xmobj.getVarRef(), new ListObj(xmobj.getContent()));
    }

    @Override
    public boolean enter(XbfFmemberRef xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.MEMBER_REF,
            xmobj.getVarRef(), new SymbolObj(xmobj.getMember()));
    }

    @Override
    public boolean enter(XbfFarrayConstructor xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_ARRAY_CONSTRUCTOR,
            xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfFstructConstructor xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_STRUCT_CONSTRUCTOR,
            xmobj.getDefModelExpr());
    }

    @Override
    public boolean enter(XbfFunctionCall xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.FUNCTION_CALL,
            xmobj.getName(), xmobj.getArguments(),
            new IntFlagObj(xmobj.getIsIntrinsic()));
    }

    @Override
    public boolean enter(XbfArguments xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.LIST,
            xmobj.getContent());
    }

    @Override
    public boolean enter(XbfVar xmobj)
    {
        return enterChild(xmobj, new SymbolObj(
            Xcode.VAR, xmobj, xmobj.getContent(), xmobj.getScope()));
    }

    @Override
    public boolean enter(XbfVarRef xmobj)
    {
        return enterAsXobjList(xmobj, Xcode.F_VAR_REF,
            xmobj.getContent());
    }

    @Override
    public boolean enter(XbfPlusExpr xmobj)
    {
        return enterBinaryExpr(Xcode.PLUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfMinusExpr xmobj)
    {
        return enterBinaryExpr(Xcode.MINUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfDivExpr xmobj)
    {
        return enterBinaryExpr(Xcode.DIV_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfMulExpr xmobj)
    {
        return enterBinaryExpr(Xcode.MUL_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfFpowerExpr xmobj)
    {
        return enterBinaryExpr(Xcode.F_POWER_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogAndExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_AND_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogEQExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_EQ_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogEQVExpr xmobj)
    {
        return enterBinaryExpr(Xcode.F_LOG_EQV_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogGEExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_GE_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogGTExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_GT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogLEExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_LE_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogLTExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_LT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogNEQExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_NEQ_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogNEQVExpr xmobj)
    {
        return enterBinaryExpr(Xcode.F_LOG_NEQV_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfLogOrExpr xmobj)
    {
        return enterBinaryExpr(Xcode.LOG_OR_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfUserBinaryExpr xmobj)
    {
        boolean r = enterBinaryExpr(Xcode.F_USER_BINARY_EXPR, xmobj);
        _xobj.add(toXobject(new StringObj(Xcode.STRING, xmobj.getName())));
        return r;
    }

    @Override
    public boolean enter(XbfLogNotExpr xmobj)
    {
        return enterUnaryExpr(Xcode.LOG_NOT_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfUnaryMinusExpr xmobj)
    {
        return enterUnaryExpr(Xcode.UNARY_MINUS_EXPR, xmobj);
    }

    @Override
    public boolean enter(XbfUserUnaryExpr xmobj)
    {
        boolean r = enterUnaryExpr(Xcode.F_USER_UNARY_EXPR, xmobj);
        _xobj.add(toXobject(new StringObj(Xcode.STRING, xmobj.getName())));
        return r;
    }
}
