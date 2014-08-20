/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import static xcodeml.util.XmLog.fatal;
import static xcodeml.util.XmLog.fatal_dump;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import exc.object.BasicType;
import exc.object.Ident;
import exc.object.LineNo;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XobjBool;
import exc.object.XobjConst;
import exc.object.XobjContainer;
import exc.object.XobjList;
import exc.object.XobjLong;
import exc.object.Xobject;
import exc.object.XobjArgs;
import exc.object.XobjectDef;
import exc.object.XobjectDefEnv;
import exc.object.XobjectFile;
import exc.object.XobjectIterator;
import exc.object.Xtype;
import exc.object.topdownXobjectIterator;
import xcodeml.IXobject;
import xcodeml.XmException;
import xcodeml.XmObj;
import xcodeml.binding.*;
import xcodeml.f.binding.IXbfBinaryExpr;
import xcodeml.f.binding.IXbfConstWithKind;
import xcodeml.f.binding.IXbfIOStatement;
import xcodeml.f.binding.IXbfRWStatement;
import xcodeml.f.binding.IXbfType;
import xcodeml.f.binding.IXbfUnaryExpr;
import xcodeml.f.binding.gen.*;
import xcodeml.util.XmBindingException;
import xcodeml.util.XmLog;
import xcodeml.util.XmXobjectToXmObjTranslator;

/**
 * Translator which translates Xobject(Xcode) to XmfObj(XcodeML/Fortran)
 */
public class XmfXobjectToXmObjTranslator
    implements XmXobjectToXmObjTranslator
{
    /** XcodeML factory */
    private IXcodeML_FFactory _factory;
    
    /** XcodeML factory method map */
    private Map<Class<?>, Method> _factoryMethodMap;
    
    public XmfXobjectToXmObjTranslator()
    {
    }

    @Override
    public XmObj translate(IXobject xobj) throws XmException
    {
        _factory = XcodeML_FFactory.getFactory();
        XmObj xmobj = null;
        xobj.setParentRecursively(null);
        
        if(xobj instanceof XobjectFile) {
            xmobj = trans((XobjectFile)xobj);
        } else if(xobj instanceof Xobject) {
            xmobj = trans((Xobject)xobj);
        } else {
            throw new XmException();
        }
        
        return xmobj;
    }

    private Boolean toBool(boolean enabled)
    {
        return enabled ? new Boolean(enabled) : null;
    }
    
    private Boolean toBool(Xobject intObj)
    {
        return (intObj != null && intObj.getInt() != 0) ? new Boolean(true) : null;
    }
    
    private XbfXcodeProgram trans(XobjectFile xobjFile)
    {
        XbfXcodeProgram xmobj = _factory.createXbfXcodeProgram();
        
        // program attributes
        xmobj.setLanguage(xobjFile.getLanguageAttribute());
        xmobj.setVersion(xobjFile.getVersion());
        xmobj.setCompilerInfo(xobjFile.getCompilerInfo());
        xmobj.setSource(xobjFile.getSourceFileName());
        xmobj.setTime(xobjFile.getTime());
        
        // type table
        xmobj.setTypeTable(transTypeTable(xobjFile.getTypeList()));
        
        // symbol table
        xmobj.setGlobalSymbols(transGlobalSymbols(xobjFile.getGlobalIdentList()));
        
        // global declarations
        addDeclForNotDeclared(xobjFile, (XobjList)xobjFile.getGlobalIdentList(), null);
        XbfGlobalDeclarations xmdecls = transGlobalDeclarations(xobjFile);
        xmobj.setGlobalDeclarations(xmdecls);
        
        // header lines
        int i = 0;
        for(XbfText text : transLines(xobjFile.getHeaderLines())) {
            xmdecls.addContent(i++, text);
        }
        
        // tailer lines
        for(XbfText text : transLines(xobjFile.getTailerLines())) {
            xmdecls.addContent(text);
        }
        
        return xmobj;
    }

    private XmObj newXmObj(Xobject xobj)
    {
        Class<?> clazz = xobj.Opcode().getXcodeML_F_Class();
        if(clazz == null)
            fatal("XcodeML class is not mapped for Xcode." + xobj.Opcode());

        if(_factory instanceof DefaultXcodeML_FFactory) {
            try {
                return (XmObj)clazz.newInstance();
            } catch(Exception e) {
                fatal(e);
            }
        }
        
        if(_factoryMethodMap == null) {
            _factoryMethodMap = new HashMap<Class<?>, Method>();
        }
        
        Method method = _factoryMethodMap.get(clazz);
        
        if(method == null) {
            String className = clazz.getName();
            // strip package name
            className = className.substring(className.lastIndexOf('.') + 1);
            String methodName = "create" + className;
            try {
                method = _factory.getClass().getMethod(methodName);
            } catch(Exception e) {
                fatal(e);
            }
            _factoryMethodMap.put(clazz, method);
        }
        
        XmObj xmobj = null;
        
        try {
            xmobj = (XmObj)method.invoke(_factory);
        } catch(Exception e) {
            fatal(e);
        }
        
        return xmobj;
    }
    
    private XmObj transOrError(Xobject xobj)
    {
        if(xobj == null)
            throw new NullPointerException("xobj");
        XmObj m = trans(xobj);
        if(m == null)
            throw new NullPointerException("m : " + xobj.toString());
        return m;
    }
    
    private String getArg0Name(Xobject x)
    {
        Xobject a = x.getArgOrNull(0);
        if(a == null)
            return null;
        return a.getName();
    }

    private XmObj trans(Xobject xobj)
    {
        if(xobj == null)
            return null;
        
        if(xobj instanceof Ident) {
            Ident i = (Ident)xobj;
            IXbStringContent sc;
            switch(i.Type().getKind()) {
            case Xtype.FUNCTION:
                sc = _factory.createXbfFfunction();
                break;
            default:
                sc = _factory.createXbfVar();
                ((XbfVar)sc).setScope(i.getScope() != null ? i.getScope().toXcodeString() : null);
                break;
            }
            sc.setContent(i.getName());
            ((IXbTypedExpr)sc).setType(i.Type().getXcodeFId());
            return (XmObj)sc;
        }
        
        XmObj xmobj = null;
        
        switch(xobj.Opcode()) {
        case F_ARRAY_INDEX: {
            XbfArrayIndex m = _factory.createXbfArrayIndex();
            xmobj = m;
            m.setDefModelExpr(transExpr(xobj.getArg(0)));
            break;
        }
        case F_INDEX_RANGE: {
            XbfIndexRange m = _factory.createXbfIndexRange();
            xmobj = m;
            Xobject lb = xobj.getArg(0);
            Xobject ub = xobj.getArg(1);
            Xobject st = xobj.getArg(2);
            Boolean isAShape = toBool(xobj.getArgOrNull(3));
            Boolean isASize = toBool(xobj.getArgOrNull(4));
            
            if(lb != null) {
                XbfLowerBound mlb = _factory.createXbfLowerBound();
                mlb.setDefModelExpr(transExpr(lb));
                m.setLowerBound(mlb);
            }
            if(ub != null) {
                XbfUpperBound mub = _factory.createXbfUpperBound();
                mub.setDefModelExpr(transExpr(ub));
                m.setUpperBound(mub);
            }
            if(st != null) {
                XbfStep mst = _factory.createXbfStep();
                mst.setDefModelExpr(transExpr(st));
                m.setStep(mst);
            }
            m.setIsAssumedShape(isAShape);
            m.setIsAssumedSize(isASize);
            break;
        }
        case FUNCTION_DEFINITION: {
            XbfFfunctionDefinition m = _factory.createXbfFfunctionDefinition();
            xmobj = m;
            XobjList symbols = (XobjList)xobj.getArg(1);
            Xobject body = xobj.getArg(3);
            XobjList decls = (XobjList)addDeclForNotDeclared((XobjList)xobj.getArg(2), symbols, body);
            m.setName(transName(xobj.getArg(0)));
            m.setSymbols(transSymbols(symbols));
            m.setDeclarations(transDeclarations(decls));
            m.setBody(transBody(body));
            break;
        }
        case VAR_DECL: {
            // (CODE name init)
            XbfVarDecl m = _factory.createXbfVarDecl();
            xmobj = m;
            m.setName(transName(xobj.getArg(0)));
            m.setValue(transValue(xobj.getArgOrNull(1)));
            break;
        }
        case FUNCTION_DECL: {
            // (CODE name () () declarations)
            XbfFfunctionDecl m = _factory.createXbfFfunctionDecl();
            xmobj = m;
            m.setName(transName(xobj.getArg(0)));
            m.setDeclarations(transDeclarations(xobj.getArgOrNull(3)));
            break;
        }
        case F_DATA_DECL: {
            XbfFdataDecl m = _factory.createXbfFdataDecl();
            xmobj = m;
            for(Xobject xseq : (XobjList)xobj) {
                XbfFdataDeclSequence mseq = new XbfFdataDeclSequence();
                m.setFdataDeclSequence(mseq);
                mseq.setVarList((XbfVarList)trans(xseq.getArg(0)));
                mseq.setValueList((XbfValueList)trans(xseq.getArg(1)));
            }
            break;
        }
        case F_VAR_LIST: {
            XbfVarList m = _factory.createXbfVarList();
            xmobj = m;
            m.setName(getArg0Name(xobj));
            for(Xobject a : (XobjList)xobj.getArg(1)) {
                m.addContent((IXbfVarListChoice)trans(a));
            }
            break;
        }
        case F_VALUE_LIST: {
            XbfValueList m = _factory.createXbfValueList();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addValue(transValue(a));
            }
            break;
        }
        case F_DO_LOOP: {
            XbfFdoLoop m = _factory.createXbfFdoLoop();
            xmobj = m;
            m.setVar((XbfVar)trans(xobj.getArg(0)));
            m.setIndexRange((XbfIndexRange)trans(xobj.getArg(1)));
            for(Xobject a : (XobjList)xobj.getArg(2)) {
                m.addValue(transValue(a));
            }
            break;
        }
        case F_BLOCK_DATA_DEFINITION: {
            XbfFblockDataDefinition m = _factory.createXbfFblockDataDefinition();
            xmobj = m;
            m.setName(getArg0Name(xobj));
            m.setSymbols(transSymbols(xobj.getArg(1)));
            m.setDeclarations(transDeclarations(xobj.getArg(2)));
            break;
        }
        case F_ENTRY_DECL: {
            XbfFentryDecl m = _factory.createXbfFentryDecl();
            xmobj = m;
            m.setName(transName(xobj.getArg(0)));
            break;
        }
        case F_EXTERN_DECL: {
            XbfExternDecl m = _factory.createXbfExternDecl();
            xmobj = m;
            m.setName(transName(xobj.getArg(0)));
            break;
        }
        case F_EQUIVALENCE_DECL: {
            XbfFequivalenceDecl m = _factory.createXbfFequivalenceDecl();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                XbfFequivalenceDeclSequence mseq = _factory.createXbfFequivalenceDeclSequence();
                m.addFequivalenceDeclSequence(mseq);
                mseq.setVarRef((XbfVarRef)trans(a.getArg(0)));
                mseq.setVarList((XbfVarList)trans(a.getArg(1)));
            }
            break;
        }
        case F_COMMON_DECL: {
            XbfFcommonDecl m = _factory.createXbfFcommonDecl();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addVarList((XbfVarList)trans(a));
            }
            break;
        }
        case F_MODULE_DEFINITION: {
            XbfFmoduleDefinition m = _factory.createXbfFmoduleDefinition();
            xmobj = m;
            m.setName(getArg0Name(xobj));
            XobjList symbols = (XobjList)xobj.getArgOrNull(1);
            XobjList decls = (XobjList)addDeclForNotDeclared((XobjList)xobj.getArgOrNull(2), symbols, null);
            m.setSymbols(transSymbols(symbols));
            m.setDeclarations(transDeclarations(decls));
            // m.setFcontainsStatement((XbfFcontainsStatement)trans(xobj.getArgOrNull(3)));
            break;
        }
        case F_MODULE_PROCEDURE_DECL: {
            XbfFmoduleProcedureDecl m = _factory.createXbfFmoduleProcedureDecl();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addName(transName(a));
            }
            break;
        }
        case F_INTERFACE_DECL: {
            // (CODE is_operator is_assignment name (LIST ... ))
            XbfFinterfaceDecl m = _factory.createXbfFinterfaceDecl();
            xmobj = m;
            m.setName(getArg0Name(xobj));
            m.setIsOperator(toBool(xobj.getArg(1)));
            m.setIsAssignment(toBool(xobj.getArg(2)));
            for(Xobject a : (XobjList)xobj.getArg(3)) {
                m.addContent((IXbfFinterfaceDeclChoice)trans(a));
            }
            break;
        }
        case F_FORMAT_DECL: {
            XbfFformatDecl m = _factory.createXbfFformatDecl();
            xmobj = m;
            m.setFormat(xobj.getArg(0).getString());
            break;
        }
        case F_NAMELIST_DECL: {
            XbfFnamelistDecl m = _factory.createXbfFnamelistDecl();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addVarList((XbfVarList)trans(a));
            }
            break;
        }
        case F_STRUCT_DECL: {
            XbfFstructDecl m = _factory.createXbfFstructDecl();
            xmobj = m;
            m.setName(transName(xobj.getArg(0)));
            break;
        }
        case F_USE_DECL: {
            XbfFuseDecl m = _factory.createXbfFuseDecl();
            xmobj = m;
            m.setName(xobj.getArg(0).getName());
            if(xobj.Nargs() > 1) {
                for(Xobject a : (XobjList)xobj.getArg(1)) {
                    m.addRename((XbfRename)trans(a));
                }
            }
            break;
        }
        case F_USE_ONLY_DECL: {
            XbfFuseOnlyDecl m = _factory.createXbfFuseOnlyDecl();
            xmobj = m;
            m.setName(xobj.getArg(0).getName());
            for(Xobject a : (XobjList)xobj.getArg(1)) {
                m.addRenamable((XbfRenamable)trans(a));
            }
            break;
        }
        case F_RENAME: {
            XbfRename m = _factory.createXbfRename();
            xmobj = m;
            m.setUseName(xobj.getArg(0).getName());
            m.setLocalName((xobj.getArgOrNull(1) != null ? xobj.getArg(1).getName() : null));
            break;
        }
        case F_RENAMABLE: {
            XbfRenamable m = _factory.createXbfRenamable();
            xmobj = m;
            m.setUseName(xobj.getArg(0).getName());
            m.setLocalName((xobj.getArgOrNull(1) != null ? xobj.getArg(1).getName() : null));
            break;
        }
        case EXPR_STATEMENT: {
            Xobject xexpr = xobj.getArg(0);
            switch(xexpr.Opcode()) {
            case FUNCTION_CALL:
                xmobj = _factory.createXbfExprStatement();
                ((XbfExprStatement)xmobj).setDefModelExpr(transExpr(xexpr));
                break;
            case ASSIGN_EXPR:
                xmobj = _factory.createXbfFassignStatement();
                ((XbfFassignStatement)xmobj).setDefModelLValue((IXbfDefModelLValueChoice)trans(xexpr.getArg(0)));
                ((XbfFassignStatement)xmobj).setDefModelExpr(transExpr(xexpr.getArg(1)));
                break;
            default:
                fatal_dump("cannot convert Xcode to XcodeML.", xobj);
            }
            break;
        }
        case F_ASSIGN_STATEMENT: {
            XbfFassignStatement m = _factory.createXbfFassignStatement();
            xmobj = m;
            m.setDefModelLValue((IXbfDefModelLValueChoice)trans(xobj.getArg(0)));
            m.setDefModelExpr(transExpr(xobj.getArg(1)));
            break;
        }
        case F_POINTER_ASSIGN_STATEMENT: {
            XbfFpointerAssignStatement m = _factory.createXbfFpointerAssignStatement();
            xmobj = m;
            m.setDefModelLValue((IXbfDefModelLValueChoice)trans(xobj.getArg(0)));
            m.setDefModelExpr(transExpr(xobj.getArg(1)));
            break;
        }
        case F_DO_STATEMENT: {
            XbfFdoStatement m = _factory.createXbfFdoStatement();
            xmobj = m;
            m.setConstructName(getArg0Name(xobj));
            Xobject var = xobj.getArgOrNull(1);
            Xobject idxRange = xobj.getArgOrNull(2);
            if(var != null) {
                XbfFdoStatementSequence mseq = _factory.createXbfFdoStatementSequence();
                mseq.setVar((XbfVar)trans(var));
                mseq.setIndexRange((XbfIndexRange)trans(idxRange));
                m.setFdoStatementSequence(mseq);
            }
            m.setBody(transBody(xobj.getArg(3)));
            break;
        }
        case F_DO_WHILE_STATEMENT: {
            XbfFdoWhileStatement m = _factory.createXbfFdoWhileStatement();
            xmobj = m;
            m.setConstructName(getArg0Name(xobj));
            m.setCondition(transCondition(xobj.getArg(1)));
            m.setBody(transBody(xobj.getArg(2)));
            break;
        }
        case WHILE_STATEMENT: {
            XbfFdoWhileStatement m = _factory.createXbfFdoWhileStatement();
            xmobj = m;
            m.setCondition(transCondition(xobj.getArg(0)));
            m.setBody(transBody(xobj.getArgOrNull(1)));
            break;
        }
        case F_SELECT_CASE_STATEMENT: {
            XbfFselectCaseStatement m = _factory.createXbfFselectCaseStatement();
            xmobj = m;
            m.setConstructName(getArg0Name(xobj));
            m.setValue(transValue(xobj.getArg(1)));
            for(Xobject a : (XobjList)xobj.getArg(2)) {
                m.addFcaseLabel((XbfFcaseLabel)trans(a));
            }
            break;
        }
        case F_CASE_LABEL: {
            XbfFcaseLabel m = _factory.createXbfFcaseLabel();
            xmobj = m;
            m.setConstructName(getArg0Name(xobj));
            XobjList values = (XobjList)xobj.getArg(1);
            if(values != null) {
                for(Xobject a : values) {
                    m.addContent((IXbfFcaseLabelChoice)trans(a));
                }
            }
            m.setBody(transBody(xobj.getArg(2)));
            break;
        }
        case F_WHERE_STATEMENT: {
            XbfFwhereStatement m = _factory.createXbfFwhereStatement();
            xmobj = m;
            m.setCondition(transCondition(xobj.getArg(1)));
            m.setThen(transThen(xobj.getArgOrNull(2)));
            Xobject xelse = xobj.getArgOrNull(3);
            if(xelse != null && xelse.Nargs() > 0)
                m.setElse(transElse(xelse));
            break;
        }
        case F_IF_STATEMENT: {
            XbfFifStatement m = _factory.createXbfFifStatement();
            xmobj = m;
            m.setConstructName(getArg0Name(xobj));
            m.setCondition(transCondition(xobj.getArg(1)));
            m.setThen(transThen(xobj.getArgOrNull(2)));
            Xobject xelse = xobj.getArgOrNull(3);
            if(xelse != null && xelse.Nargs() > 0)
                m.setElse(transElse(xelse));
            break;
        }
        case IF_STATEMENT: {
            XbfFifStatement m = _factory.createXbfFifStatement();
            xmobj = m;
            m.setCondition(transCondition(xobj.getArg(0)));
            m.setThen(transThen(xobj.getArgOrNull(1)));
            Xobject xelse = xobj.getArgOrNull(2);
            if(xelse != null && xelse.Nargs() > 0)
                m.setElse(transElse(xelse));
            break;
        }
        case F_CYCLE_STATEMENT: {
            XbfFcycleStatement m = _factory.createXbfFcycleStatement();
            m.setConstructName(getArg0Name(xobj));
            xmobj = m;
            break;
        }
        case F_EXIT_STATEMENT: {
            XbfFexitStatement m = _factory.createXbfFexitStatement();
            m.setConstructName(getArg0Name(xobj));
            xmobj = m;
            break;
        }
        case F_CONTINUE_STATEMENT:
        case RETURN_STATEMENT: {
            xmobj = newXmObj(xobj);
            break;
        }
        case GOTO_STATEMENT: {
            XbfGotoStatement m = _factory.createXbfGotoStatement();
            xmobj = m;
            m.setLabelName(getArg0Name(xobj));
            Xobject value = xobj.getArgOrNull(1);
            Xobject params = xobj.getArgOrNull(2);
            if(value != null) {
                XbfGotoStatementSequence mseq = _factory.createXbfGotoStatementSequence();
                mseq.setValue(transValue(value));
                mseq.setParams(transParams((XobjList)params));
                m.setGotoStatementSequence(mseq);
            }
            break;
        }
        case STATEMENT_LABEL: {
            XbfStatementLabel m = _factory.createXbfStatementLabel();
            xmobj = m;
            m.setLabelName(getArg0Name(xobj));
            break;
        }
        case F_CONTAINS_STATEMENT: {
            XbfFcontainsStatement m = _factory.createXbfFcontainsStatement();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addFfunctionDefinition((XbfFfunctionDefinition)trans(a));
            }
            break;
        }
        case PRAGMA_LINE: {
            XbfFpragmaStatement m = _factory.createXbfFpragmaStatement();
            xmobj = m;
            m.setContent(xobj.getArg(0).getString());
            break;
        }
        case TEXT: {
            XbfText m = _factory.createXbfText();
            xmobj = m;
            m.setContent(xobj.getArg(0).getString());
            break;
        }
        case F_ALLOCATE_STATEMENT: {
            XbfFallocateStatement m = _factory.createXbfFallocateStatement();
            xmobj = m;
            //m.setStatName("alloc");// m.setStatName(getArg0Name(xobj));
            for(Xobject a : (XobjList)xobj) {
	      if(a == null) continue;
	      if(a.Opcode() == Xcode.LIST){
		for(Xobject aa : (XobjList)a)
		  m.addAlloc((XbfAlloc)trans(aa));
	      } else 
                m.addAlloc((XbfAlloc)trans(a));
            }
            break;
        }
        case F_DEALLOCATE_STATEMENT: {
            XbfFdeallocateStatement m = _factory.createXbfFdeallocateStatement();
            xmobj = m;
            m.setStatName(getArg0Name(xobj));
            for(Xobject a : (XobjList)xobj.getArg(1)) {
                m.addAlloc((XbfAlloc)trans(a));
            }
            break;
        }
        case F_NULLIFY_STATEMENT: {
            XbfFnullifyStatement m = _factory.createXbfFnullifyStatement();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addAlloc((XbfAlloc)trans(a));
            }
            break;
        }
        case F_ALLOC: {
            XbfAlloc m = _factory.createXbfAlloc();
            xmobj = m;
            m.setContent((IXbfAllocChoice1)trans(xobj.getArg(0)));
	    XobjArgs args;
	    /* (F_ALLOC v (LIST range ...)) or (F_ALLOC v range range) */
	    if(xobj.getArg(1).Opcode() == Xcode.LIST)
	      args = xobj.getArg(1).getArgs();
	    else
	      args = xobj.getArgs().nextArgs();
            if(args != null) {
	      for(; args != null; args = args.nextArgs()){
		Xobject a = args.getArg();
                    m.addDefModelArraySubscript((IXbfDefModelArraySubscriptChoice)trans(a));
                }
            }
            break;
        }
        case F_OPEN_STATEMENT:
        case F_CLOSE_STATEMENT:
        case F_END_FILE_STATEMENT:
        case F_REWIND_STATEMENT:
        case F_BACKSPACE_STATEMENT: {
            xmobj = newXmObj(xobj);
            IXbfIOStatement m = (IXbfIOStatement)xmobj;
            m.setNamedValueList((XbfNamedValueList)trans(xobj.getArgOrNull(0)));
            break;
        }
        case F_NAMED_VALUE_LIST: {
            XbfNamedValueList m = _factory.createXbfNamedValueList();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addNamedValue((XbfNamedValue)trans(a));
            }
            break;
        }
        case F_NAMED_VALUE: {
            XbfNamedValue m = _factory.createXbfNamedValue();
            xmobj = m;
            m.setName(getArg0Name(xobj));
            Xobject value = xobj.getArg(1);
            if(value.Opcode() == Xcode.STRING) {
                m.setValue(value.getString());
            } else {
                m.setDefModelExpr(transExpr(value));
            }
            break;
        }
        case F_READ_STATEMENT:
        case F_WRITE_STATEMENT:
        case F_INQUIRE_STATEMENT: {
            xmobj = newXmObj(xobj);
            IXbfRWStatement m = (IXbfRWStatement)xmobj;
            m.setNamedValueList((XbfNamedValueList)trans(xobj.getArg(0)));
            m.setValueList((XbfValueList)trans(xobj.getArg(1)));
            break;
        }
        case F_PRINT_STATEMENT: {
            XbfFprintStatement m = _factory.createXbfFprintStatement();
            xmobj = m;
            m.setFormat(xobj.getArg(0).getString());
            m.setValueList((XbfValueList)trans(xobj.getArgOrNull(1)));
            break;
        }
        case F_PAUSE_STATEMENT: {
            XbfFpauseStatement m = _factory.createXbfFpauseStatement();
            xmobj = m;
            Xobject code = xobj.getArgOrNull(0);
            Xobject msg = xobj.getArgOrNull(1);
            if(code != null)
                m.setCode(code.getString());
            if(msg != null)
                m.setMessage(msg.getString());
            break;
        }
        case F_STOP_STATEMENT: {
            XbfFstopStatement m = _factory.createXbfFstopStatement();
            xmobj = m;
            Xobject code = xobj.getArgOrNull(0);
            Xobject msg = xobj.getArgOrNull(1);
            if(code != null)
                m.setCode(code.getString());
            if(msg != null)
                m.setMessage(msg.getString());
            break;
        }
        case VAR: {
            XbfVar m = _factory.createXbfVar();
            xmobj = m;
            m.setContent(xobj.getName());
            m.setScope(xobj.getScope() != null ? xobj.getScope().toXcodeString() : null);
            break;
        }
        case F_VAR_REF: {
            XbfVarRef m = _factory.createXbfVarRef();
            xmobj = m;
            m.setContent((IXbfVarRefChoice)trans(xobj.getArg(0)));
            break;
        }
        case F_ARRAY_REF: {
            XbfFarrayRef m = _factory.createXbfFarrayRef();
            xmobj = m;
            m.setVarRef((XbfVarRef)trans(xobj.getArg(0)));
            for(Xobject a : (XobjList)xobj.getArg(1)) {
                m.addContent((IXbfFarrayRefChoice)trans(a));
            }
            break;
        }
        case F_USER_UNARY_EXPR:
        case LOG_NOT_EXPR:
        case UNARY_MINUS_EXPR: {
            xmobj = newXmObj(xobj);
            IXbfUnaryExpr m = (IXbfUnaryExpr)xmobj;
            m.setDefModelExpr(transExpr(xobj.getArg(0)));
            
            if(xobj.Opcode() == Xcode.F_USER_UNARY_EXPR) {
                ((XbfUserUnaryExpr)m).setName(xobj.getArg(1).getString());
            }
            break;
        }
        case PLUS_EXPR:
        case MINUS_EXPR:
        case MUL_EXPR:
        case DIV_EXPR:
        case F_POWER_EXPR:
        case LOG_AND_EXPR:
        case LOG_EQ_EXPR:
        case F_LOG_EQV_EXPR:
        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case LOG_NEQ_EXPR:
        case F_LOG_NEQV_EXPR:
        case LOG_OR_EXPR:
        case F_CONCAT_EXPR:
        case F_USER_BINARY_EXPR: {
            xmobj = newXmObj(xobj);
            IXbfBinaryExpr m = (IXbfBinaryExpr)xmobj;
            m.setDefModelExpr1(transExpr(xobj.getArg(0)));
            m.setDefModelExpr2(transExpr(xobj.getArg(1)));
            
            if(xobj.Opcode() == Xcode.F_USER_BINARY_EXPR) {
                ((XbfUserBinaryExpr)m).setName(xobj.getArg(2).getString());
            }
            break;
        }
        case F_CHARACTER_REF: {
            XbfFcharacterRef m = _factory.createXbfFcharacterRef();
            xmobj = m;
            m.setVarRef((XbfVarRef)trans(xobj.getArg(0)));
            m.setIndexRange((XbfIndexRange)trans(xobj.getArg(1)));
            break;
        }
        case INT_CONSTANT:
        case LONGLONG_CONSTANT:
        case FLOAT_CONSTANT:
        case F_LOGICAL_CONSTATNT:
        case F_CHARACTER_CONSTATNT:
        case STRING_CONSTANT: {
            xmobj = newXmObj(xobj);
            IXbfConstWithKind m = (IXbfConstWithKind)xmobj;
            m.setContent(getConstContent(xobj));
            m.setKind(((XobjConst)xobj).getFkind());
            break;
        }
        case F_COMPLEX_CONSTATNT: {
            XbfFcomplexConstant m = _factory.createXbfFcomplexConstant();
            xmobj = m;
            m.setDefModelExpr1(transExpr(xobj.getArg(0)));
            m.setDefModelExpr2(transExpr(xobj.getArg(1)));
            break;
        }
        case FUNC_ADDR: {
            XbfFfunction m = _factory.createXbfFfunction();
            xmobj = m;
            m.setContent(xobj.getName());
            break;
        }
        case FUNCTION_CALL: {
            XbfFunctionCall m = _factory.createXbfFunctionCall();
            xmobj = m;
            m.setName(transName(xobj.getArg(0)));
            if(xobj.getArg(1) != null) {
                XbfArguments margs = _factory.createXbfArguments();
                for(Xobject a : (XobjList)xobj.getArg(1)) {
                    margs.addContent((IXbfArgumentsChoice)trans(a));
                }
                m.setArguments(margs);
            }
            m.setIsIntrinsic(toBool(xobj.getArgOrNull(2)));
            break;
        }
        case MEMBER_REF: {
            XbfFmemberRef m = _factory.createXbfFmemberRef();
            xmobj = m;
            m.setVarRef((XbfVarRef)trans(xobj.getArg(0)));
            m.setMember(xobj.getArg(1).getName());
            break;
        }
        case F_ARRAY_CONSTRUCTOR: {
            XbfFarrayConstructor m = _factory.createXbfFarrayConstructor();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addDefModelExpr(transExpr(a));
            }
            break;
        }
        case F_STRUCT_CONSTRUCTOR: {
            XbfFstructConstructor m = _factory.createXbfFstructConstructor();
            xmobj = m;
            for(Xobject a : (XobjList)xobj) {
                m.addDefModelExpr(transExpr(a));
            }
            break;
        }
        case F_VALUE: {
            xmobj = transValue(xobj);
            break;
        }
        case NULL:
            return null;
        default:
            fatal_dump("cannot convert Xcode to XcodeML.", xobj);
        }
        
        if(xmobj instanceof IXbTypedExpr) {
            if(xobj.Type() != null) {
                String tid = xobj.Type().getXcodeFId();
//                 if(tid == null || tid.equals("null"))
//                    XmLog.fatal("type is null");
                ((IXbTypedExpr)xmobj).setType(tid);
            }
        }
        
        if(xmobj instanceof IXbLineNo) {
            LineNo lineno = xobj.getLineNo();
            if(lineno != null) {
                ((IXbLineNo)xmobj).setFile(lineno.fileName());
                ((IXbLineNo)xmobj).setLineno(Integer.toString(lineno.lineNo()));
            }
        }
        
        return xmobj;
    }
    
    private IXbfDefModelExprChoice transExpr(Xobject xobj)
    {
        return (IXbfDefModelExprChoice)trans(xobj);
    }

    private XbfValue transValue(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbfValue mval = _factory.createXbfValue();
        mval.setDefModelExpr(transExpr(xobj.getArg(0)));
        IXbfDefModelExprChoice mrcExpr = transExpr(xobj.getArgOrNull(1));
        if(mrcExpr != null) {
            XbfRepeatCount mrc = _factory.createXbfRepeatCount();
            mrc.setDefModelExpr(mrcExpr);
            mval.setRepeatCount(mrc);
        }
        return mval;
    }
    
    private XobjContainer addDeclForNotDeclared(XobjContainer declList, XobjList identList, Xobject body)
    {
        if(identList == null)
            return null;
        
        // collect identifiers which are set to 'delayed decl'
        if(body != null && body instanceof XobjList) {
            XobjectIterator i = new topdownXobjectIterator(body);
            for(i.init(); !i.end(); i.next()) {
                Xobject a = i.getXobject();
                if(a == null || !a.isDelayedDecl())
                    continue;
                String name = a.getName();
                if(identList.hasIdent(name))
                    continue;
                
                Ident id = Ident.Fident(name, a.Type(), a.isToBeFcommon(), true, null);
                identList.add(id);
            }
        }
        
        XobjList addDeclList = Xcons.List();

        // add declaration
        for(Xobject a : identList) {
            Ident id = (Ident)a;
            if(id.isDeclared())
                continue;
            
            String name = id.getName();
            if(declList instanceof XobjList) {
                boolean exists = false;
                for(Xobject decl : (XobjList)declList) {
                    if(decl != null && decl.Opcode() == Xcode.VAR_DECL &&
                        decl.getArg(0).getName().equals(name)) {
                        exists = true;
                        break;
                    }
                }
                if(exists)
                    continue;
            }
            if(id.getStorageClass().isVarOrFunc()) {
                addDeclList.add(Xcons.List(Xcode.VAR_DECL,
                    Xcons.Symbol(Xcode.IDENT, id.Type(), name), id.getFparamValue()));
            }
            
            if(id.isToBeFcommon()) {
                Xobject cmnDecl = Xcons.List(Xcode.F_COMMON_DECL,
                    Xcons.List(Xcode.F_VAR_LIST, Xcons.Symbol(Xcode.IDENT, id.Type(), name),
                        Xcons.List(Xcode.LIST, Xcons.FvarRef(id))));
                addDeclList.add(cmnDecl);
            }
        }
        
        if(!addDeclList.isEmpty()) {
            if(declList == null)
                declList = addDeclList;
            else {
                addDeclList.reverse();
                for(Xobject a : addDeclList) {
                    declList.insert(a);
                }
            }
        }

        if(declList instanceof XobjList) {
            // redorder var decls by dependency.
            new DeclSorter(identList).sort((XobjList)declList);
        }
        
        return declList;
    }
    
    private static final String PROP_KEY_IDSET = "DeclComparator.idSet";
    private static final String PROP_KEY_DEPSET = "DeclComparator.depSet";

    /**
     * sort declarations by dependency order.
     */
    static class DeclSorter
    {
        private XobjList ident_list;
        
        public DeclSorter(XobjList ident_list)
        {
            this.ident_list = ident_list;
        }
        
        public void sort(XobjList declList)
        {
            Map<String, Xobject> declMap = new HashMap<String, Xobject>();
            List<Xobject> headDeclList = new ArrayList<Xobject>();
            List<Xobject> tailDeclList = new ArrayList<Xobject>();
            
            for(Xobject decl : declList) {
                collectDependName(decl);
                switch(decl.Opcode()) {
                case VAR_DECL:
                    declMap.put(decl.getArg(0).getName(), decl);
                    break;
                case F_STRUCT_DECL:
                    declMap.put("$" + decl.getArg(0).getName(), decl);
                    break;
                case F_USE_DECL:
                case F_USE_ONLY_DECL:
                    headDeclList.add(decl);
                    break;
                default:
                    tailDeclList.add(decl);
                    break;
                }
            }
            
            for(Xobject decl : declMap.values()) {
                Set<String> idSet = getIdSet(decl);
                if(idSet == null)
                    throw new IllegalStateException(decl.toString());
                decl.remProp(PROP_KEY_IDSET);
                Set<Xobject> depSet = getDepSet(decl);
                if(depSet == null) {
                    depSet = new HashSet<Xobject>();
                    decl.setProp(PROP_KEY_DEPSET, depSet);
                }
                for(String n : idSet) {
                    Xobject depDecl = declMap.get(n);
                    if(depDecl == null)
                        continue;
                    depSet.add(depDecl);
                }
            }
            
            declList.clear();

            while(true) {
                int declMapSize = declMap.size();
                
                for(Iterator<Xobject> ite = declMap.values().iterator(); ite.hasNext(); ) {
                    Xobject decl = ite.next();
                    Set<Xobject> depSet = getDepSet(decl);
                    if(depSet.isEmpty()) {
                        ite.remove();
                        declList.add(decl);
                        decl.remProp(PROP_KEY_DEPSET);
                        for(Xobject d : declMap.values()) {
                            getDepSet(d).remove(decl);
                        }
                    }
                }
                
                if(declMapSize == declMap.size()) {
                    for(Xobject d : declMap.values())
                        declList.add(d);
                    break;
                }
                
                if(declMap.isEmpty())
                    break;
            }

            ListIterator<Xobject> ite = headDeclList.listIterator();
            for(; ite.hasNext(); ite.next());
            for(; ite.hasPrevious(); ) {
                Xobject x = ite.previous();
                declList.insert(x);
            }
            
            for(Xobject x : tailDeclList)
                declList.add(x);
        }
        
        @SuppressWarnings("unchecked")
		private Set<String> getIdSet(Xobject x)
        {
    		return (Set<String>)x.getProp(PROP_KEY_IDSET);
    	}
        
        @SuppressWarnings("unchecked")
       private Set<Xobject> getDepSet(Xobject x)
        {
            return (Set<Xobject>)x.getProp(PROP_KEY_DEPSET);
        }

        private void collectDependName(Xobject decl)
        {
            switch(decl.Opcode()) {
            case VAR_DECL:
            case F_STRUCT_DECL:
                break;
            default:
                return;
            }
            
            Set<String> idSet = getIdSet(decl);
            if(idSet != null)
                return;
            idSet = new HashSet<String>();
            decl.setProp(PROP_KEY_IDSET, idSet);
            
            Ident id = ident_list.find(decl.getArg(0).getName(),
                (decl.Opcode() == Xcode.VAR_DECL) ? IXobject.FINDKIND_VAR : IXobject.FINDKIND_TAGNAME);
            Xtype t = (id != null) ? id.Type() : null;
            _collectDependName(t, idSet);
            
            if(decl.Opcode() == Xcode.VAR_DECL) {
                _collectDependName(decl.getArgOrNull(1), idSet);
            } else {
                //remove dependency to self
                idSet.remove("$" + id.getName());
            }
        }
        
        private void _collectDependName(Xobject x, Set<String> idSet)
        {
            if(x == null)
                return;
            
            if(x.isVarRef()) {
                idSet.add(x.getName());
            } else if(x instanceof XobjList) {
                for(Xobject a : (XobjList)x)
                    _collectDependName(a, idSet);
            }
        }

        private void _collectDependName(Xtype t, Set<String> idSet)
        {
            if(t == null)
                return;
            
            switch(t.getKind()) {
            case Xtype.BASIC:
                if(t.copied != null) {
                    _collectDependName(t.copied, idSet);
                } else {
                    _collectDependName(t.getFkind(), idSet);
                    _collectDependName(t.getFlen(), idSet);
                }
                break;
            case Xtype.F_ARRAY:
                _collectDependName(t.getRef(), idSet);
                for(Xobject s : t.getFarraySizeExpr())
                    _collectDependName(s, idSet);
                break;
            case Xtype.STRUCT: {
                    Ident typeName = ident_list.getStructTypeName(t);
                    if(typeName != null)
                        idSet.add("$" + typeName.getName());
                    for(Xobject a : t.getMemberList()) {
                        _collectDependName(a.Type(), idSet);
                        _collectDependName(((Ident)a).getValue(), idSet);
                    }
                }
                break;
            }
        }
    }
    
    private List<XbfText> transLines(List<String> lines)
    {
        ArrayList<XbfText> textList = new ArrayList<XbfText>(lines != null ? lines.size() : 0);
        
        if(lines == null)
            return textList;
        
        for(String line : lines) {
            if(line == null)
                continue;
            String[] splines = line.split("\n");
            for(String spline : splines) {
                XbfText text = new XbfText();
                text.setContent(spline);
                textList.add(text);
            }
        }
        
        return textList;
    }
    
    private XbfTypeTable transTypeTable(List<Xtype> xtypeList)
    {
        XbfTypeTable xmobj = _factory.createXbfTypeTable();
        if(xtypeList != null) {
            for(Xtype xtype : xtypeList) {
                xmobj.addContent(transType(xtype));
            }
        }
        return xmobj;
    }
    
    private void setBasicTypeFlags(XbfFbasicType bt, Xtype type)
    {
        bt.setIsAllocatable(toBool(type.isFallocatable()));
        bt.setIsOptional(toBool(type.isFoptional()));
        bt.setIsParameter(toBool(type.isFparameter()));
        bt.setIsPointer(toBool(type.isFpointer()));
        bt.setIsPrivate(toBool(type.isFprivate()));
        bt.setIsPublic(toBool(type.isFpublic()));
        bt.setIsSave(toBool(type.isFsave()));
        bt.setIsTarget(toBool(type.isFtarget()));
        bt.setIsCrayPointer(toBool(type.isFcrayPointer()));
        if(type.isFintentIN())
            bt.setIntent(XbfFbasicType.INTENT_IN);
        if(type.isFintentOUT())
            bt.setIntent(XbfFbasicType.INTENT_OUT);
        if(type.isFintentINOUT())
            bt.setIntent(XbfFbasicType.INTENT_INOUT);
    }

    private IXbfTypeTableChoice transType(Xtype type)
    {
        IXbfType xmtype = null;
        
        if(type.copied != null) {
            XbfFbasicType t = _factory.createXbfFbasicType();
            t.setRef(type.copied.getXcodeFId());
            setBasicTypeFlags(t, type);
            xmtype = t;
        } else {
            switch(type.getKind()) {
            case Xtype.BASIC:
            case Xtype.F_ARRAY: {
                XbfFbasicType t = _factory.createXbfFbasicType();
                
                if(type.getKind() == Xtype.BASIC) {
                    t.setRef(BasicType.getTypeInfo(type.getBasicType()).fname);
                    t.setKind(transKind(type.getFkind()));
                    t.setContent(transLen(type));
                } else {
                    t.setRef(type.getRef().getXcodeFId());
                    XbfDefModelArraySubscriptSequence1 msubs = new XbfDefModelArraySubscriptSequence1();
                    t.setContent(msubs);
                    for(Xobject sizeExpr : type.getFarraySizeExpr()) {
                        msubs.addDefModelArraySubscript((IXbfDefModelArraySubscriptChoice)trans(sizeExpr));
                    }
                }

                setBasicTypeFlags(t, type);
                xmtype = t;
                break;
            }
                
            case Xtype.STRUCT: {
                XbfFstructType t = _factory.createXbfFstructType();
                t.setSymbols(transSymbols(type.getMemberList()));
                
                t.setIsInternalPrivate(toBool(type.isFinternalPrivate()));
                t.setIsPrivate(toBool(type.isFprivate()));
                t.setIsPublic(toBool(type.isFpublic()));
                t.setIsSequence(toBool(type.isFsequence()));
                xmtype = t;
                break;
            }
            
            case Xtype.FUNCTION: {
                XbfFfunctionType t = _factory.createXbfFfunctionType();
                t.setResultName(type.getFuncResultName());
                t.setReturnType(type.getRef().getXcodeFId());
                t.setParams(transParams((XobjList)type.getFuncParam()));
                t.setIsExternal(toBool(type.isFexternal()));
                t.setIsInternal(toBool(type.isFinternal()));
                t.setIsIntrinsic(toBool(type.isFintrinsic()));
                t.setIsPrivate(toBool(type.isFprivate()));
                t.setIsPublic(toBool(type.isFpublic()));
                t.setIsProgram(toBool(type.isFprogram()));
                t.setIsRecursive(toBool(type.isFrecursive()));
                xmtype = t;
                break;
            }
            
            default:
                fatal("cannot convert type_kind:" + Xtype.getKindName(type.getKind()));
            }
        }

        xmtype.setType(type.getXcodeFId());
        
        return (IXbfTypeTableChoice)xmtype;
    }
    
    private XbfKind transKind(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbfKind kind = _factory.createXbfKind();
        kind.setDefModelExpr(transExpr(xobj));
        return kind;
    }
    
    private XbfLen transLen(Xtype type)
    {
        Xobject xobj = type.getFlen();
        if(xobj == null)
            return null;
        XbfLen len = _factory.createXbfLen();
        if(!type.isFlenVariable())
            len.setDefModelExpr(transExpr(xobj));
        return len;
    }
    
    private XbfParams transParams(XobjList paramList)
    {
        if(paramList == null || paramList.isEmpty())
            return null;
        XbfParams xmparams = _factory.createXbfParams();
        for(Xobject param : paramList) {
            xmparams.addName(transName(param));
        }
        return xmparams;
    }
    
    private XbfName transName(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbfName xmname = _factory.createXbfName();
        if(xobj.Type() != null)
            xmname.setType(xobj.Type().getXcodeFId());
        xmname.setContent(xobj.getName());
        return xmname;
    }
    
    private XbfGlobalSymbols transGlobalSymbols(Xobject xobj)
    {
        XobjList identList = (XobjList)xobj;
        XbfGlobalSymbols xmobj = _factory.createXbfGlobalSymbols();
        if(identList != null) {
            for(Xobject ident : identList) {
                xmobj.addId(transIdent((Ident)ident));
            }
        }
        return xmobj;
    }
    
    private XbfSymbols transSymbols(Xobject xobj)
    {
        XobjList identList = (XobjList)xobj;
        XbfSymbols xmobj = _factory.createXbfSymbols();
        if(identList != null) {
            for(Xobject ident : identList) {
                xmobj.addId(transIdent((Ident)ident));
            }
        }
        return xmobj;
    }

    private XbfGlobalDeclarations transGlobalDeclarations(XobjectDefEnv defList)
    {
        XbfGlobalDeclarations xmgdecls = _factory.createXbfGlobalDeclarations();
        for(XobjectDef def : defList) {
            IXbfGlobalDeclarationsChoice xmdef = (IXbfGlobalDeclarationsChoice)transDef(def);
            if(xmdef != null)
                xmgdecls.addContent(xmdef);
        }
        return xmgdecls;
    }

    private XbfDeclarations transDeclarations(Xobject xobj)
    {
        XbfDeclarations xmdecls = _factory.createXbfDeclarations();
        if(xobj != null) {
            for(Xobject a : (XobjList)xobj) {
                if(a == null)
                    continue;
                XmObj m = trans(a);
                if(m == null)
                    continue;
                xmdecls.addContent((IXbfDeclarationsChoice)m);
            }
        }
        return xmdecls;
    }
    
    private XmObj transDef(XobjectDef def)
    {
        if(def == null)
            throw new XmBindingException(null, "def is null");
        Xobject defObj = def.getDef();
        if(defObj == null)
            return null;
        XmObj xmobj = transOrError(defObj);
        
        if(def.hasChildren()) {
            XbfFcontainsStatement xmcont = _factory.createXbfFcontainsStatement();
            for(XobjectDef childDef : def.getChildren()) {
                XmObj childXmobj = transDef(childDef);
                xmcont.addFfunctionDefinition((XbfFfunctionDefinition)childXmobj);
            }
            if(xmobj instanceof XbfFfunctionDefinition) {
                ((XbfFfunctionDefinition)xmobj).getBody().addDefModelStatement(xmcont);
            } else if(xmobj instanceof XbfFmoduleDefinition) {
                ((XbfFmoduleDefinition)xmobj).setFcontainsStatement(xmcont);
            } else {
                throw new XmBindingException(null, xmobj.getClass().getName());
            }
        }
        
        return xmobj;
    }

    private XbfId transIdent(Ident ident)
    {
        if(ident == null)
            return null;
        XbfId xmid = _factory.createXbfId();
        
        // type
        if(ident.Type() != null)
            xmid.setType(ident.Type().getXcodeFId());
        
        // name
        XbfName xmname = transName(ident);
        xmname.setType(null);
        xmid.setName(xmname);
        
        // sclass
        if(ident.getStorageClass() != null)
            xmid.setSclass(ident.getStorageClass().toXcodeString());

        // value
        Xobject val = ident.getValue();
        if (val != null && val.Opcode() == Xcode.F_VALUE) {
            xmid.setValue(transValue(val));
        }

        return xmid;
    }
    
    private void addToBody(XbfBody xmbody, Xobject xobj)
    {
        if(xobj == null)
            return;
        switch(xobj.Opcode()) {
        case F_STATEMENT_LIST:
        case LIST:
            for(Xobject a : (XobjList)xobj) {
                addToBody(xmbody, a);
            }
            break;
        case COMPOUND_STATEMENT:
            throw new IllegalArgumentException();
        default: {
                IXbfDefModelStatementChoice xmstmt = (IXbfDefModelStatementChoice)trans(xobj);
                if(xmstmt != null)
                    xmbody.addDefModelStatement(xmstmt);
            }
            break;
        }
    }

    private XbfBody transBody(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbfBody xmbody = _factory.createXbfBody();
        addToBody(xmbody, xobj);
        return xmbody;
    }
    
    private XbfCondition transCondition(Xobject xobj)
    {
        XbfCondition xmcond = _factory.createXbfCondition();
        xmcond.setDefModelExpr(transExpr(xobj));
        return xmcond;
    }
    
    private XbfThen transThen(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbfThen xmthen = _factory.createXbfThen();
        xmthen.setBody(transBody(xobj));
        return xmthen;
    }

    private XbfElse transElse(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbfElse xmelse = _factory.createXbfElse();
        xmelse.setBody(transBody(xobj));
        return xmelse;
    }

    private String getConstContent(Xobject xobj)
    {
        switch(xobj.Opcode()) {
        case INT_CONSTANT:
            return Integer.toString(xobj.getInt());
        case LONGLONG_CONSTANT:
            return ((XobjLong)xobj).getBigInteger().toString();
        case FLOAT_CONSTANT:
            return xobj.getFloatString();
        case F_CHARACTER_CONSTATNT:
        case STRING_CONSTANT:
            return xobj.getString();
        case F_LOGICAL_CONSTATNT:
            return ((XobjBool)xobj).getBoolValue() ? ".TRUE." : ".FALSE.";
        }
        throw new UnsupportedOperationException("not constant : " + xobj.OpcodeName());
    }
}
