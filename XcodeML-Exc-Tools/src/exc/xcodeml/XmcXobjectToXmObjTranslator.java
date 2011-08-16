/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import exc.object.BasicType;
import exc.object.Ident;
import exc.object.StorageClass;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XobjList;
import exc.object.Xobject;
import exc.object.XobjectDef;
import exc.object.XobjectDefEnv;
import exc.object.XobjectFile;
import exc.object.Xtype;
import xcodeml.IXobject;
import xcodeml.XmException;
import xcodeml.XmObj;
import xcodeml.binding.IXbLineNo;
import xcodeml.binding.IXbStringContent;
import xcodeml.c.binding.*;
import xcodeml.c.binding.gen.*;
import xcodeml.util.XmBindingException;
import xcodeml.util.XmXobjectToXmObjTranslator;

import static xcodeml.util.XmLog.fatal;
import static xcodeml.util.XmLog.fatal_dump;

/**
 * Translator which translates Xobject(Xcode) to XmcObj(XcodeML/C)
 */
public class XmcXobjectToXmObjTranslator
    implements XmXobjectToXmObjTranslator
{
    private IXcodeML_CFactory _factory;
    
    private Map<Class<?>, Method> _factoryMethodMap;
    
    private static final String TRUE_STR = "1";
    
    public XmcXobjectToXmObjTranslator()
    {
    }
    
    @Override
    public XmObj translate(IXobject xobj) throws XmException
    {
        _factory = XcodeML_CFactory.getFactory();
        XmObj xmobj = null;
        
        if(xobj instanceof XobjectFile) {
            xmobj = trans((XobjectFile)xobj);
        } else if(xobj instanceof Xobject) {
            xmobj = trans((Xobject)xobj);
        } else {
            throw new XmException();
        }
        
        return xmobj;
    }
    
    private XbcXcodeProgram trans(XobjectFile xobjFile)
    {
        XbcXcodeProgram xmobj = _factory.createXbcXcodeProgram();
        
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
        XobjList declList = getDeclForNotDeclared((XobjList)xobjFile.getGlobalIdentList());
        if(declList != null) {
            declList.reverse();
            for(Xobject a : declList)
                xobjFile.insert(a);
        }
        XbcGlobalDeclarations xmdecls = transGlobalDeclarations(xobjFile);
        xmobj.setGlobalDeclarations(xmdecls);
        
        // header lines
        int i = 0;
        for(XbcText text : transLines(xobjFile.getHeaderLines())) {
            xmdecls.addContent(i++, text);
        }
        
        // tailer lines
        for(XbcText text : transLines(xobjFile.getTailerLines())) {
            xmdecls.addContent(text);
        }
        
        return xmobj;
    }
    
    private XobjList getDeclForNotDeclared(XobjList identList)
    {
        if(identList == null)
            return null;
        
        XobjList declList = Xcons.List();
        
        for(Xobject a : identList) {
            Ident id = (Ident)a;
            if(id.isDeclared() || !id.getStorageClass().isVarOrFunc())
                continue;
            Xtype t = id.Type();
            Xcode declCode = t.isFunction() ? Xcode.FUNCTION_DECL : Xcode.VAR_DECL;
            declList.add(Xcons.List(declCode, Xcons.Symbol(Xcode.IDENT, id.getName()), null));
        }
        
        if(declList.isEmpty())
            return null;
        
        return declList;
    }
    
    private XmObj newXmObj(Xobject xobj)
    {
        Class<?> clazz = xobj.Opcode().getXcodeML_C_Class();
        if(clazz == null)
            fatal("XcodeML class is not mapped for Xcode." + xobj.Opcode());

        if(_factory instanceof DefaultXcodeML_CFactory) {
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
    
    private String getTypeId(Xobject xobj)
    {
        Xtype type = xobj.Type();
        if(type == null)
            fatal("type is null : " + xobj.toString());
        return type.getXcodeCId();
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
    
    private XmObj trans(Xobject xobj)
    {
        if(xobj == null)
            return null;
        
        if(xobj instanceof Ident) {
            Ident i = (Ident)xobj;
            IXbStringContent sc;
            switch(i.Type().getKind()) {
            case Xtype.FUNCTION:
                sc = _factory.createXbcFuncAddr();
                break;
            case Xtype.ARRAY:
                sc = _factory.createXbcArrayAddr();
                ((XbcArrayAddr)sc).setScope(i.getScope() != null ? i.getScope().toXcodeString() : null);
                break;
            default:
                sc = _factory.createXbcVar();
                ((XbcVar)sc).setScope(i.getScope() != null ? i.getScope().toXcodeString() : null);
                break;
            }
            sc.setContent(i.getName());
            ((IXbcTypedExpr)sc).setType(i.Type().getXcodeCId());
            return (XmObj)sc;
        }
        
        XmObj xmobj = null;
        
        switch(xobj.Opcode()) {
        case LIST: {
                switch(xobj.Nargs()) {
                case 0:
                    return null;
                case 1:
                    return trans(xobj.getArg(0));
                default:
                    XbcCompoundStatement m = _factory.createXbcCompoundStatement();
                    XbcBody mbody = _factory.createXbcBody();
                    m.setBody(mbody);
                    m.setSymbols(_factory.createXbcSymbols());
                    m.setDeclarations(_factory.createXbcDeclarations());
                    for(Xobject a: (XobjList)xobj) {
                        IXbcStatementsChoice mstmt = (IXbcStatementsChoice)trans(a);
                        if(mstmt != null)
                            mbody.addStatements(mstmt);
                    }
                    xmobj = m;
                }
            }
            break;
        
        // constant
            
        case STRING_CONSTANT: {
                XbcStringConstant m = _factory.createXbcStringConstant();
                m.setContent(xobj.getString());
                xmobj = m;
            }
            break;
        case INT_CONSTANT: {
                XbcIntConstant m = _factory.createXbcIntConstant();
                m.setContent("" + xobj.getInt());
                xmobj = m;
            }
            break;
        case FLOAT_CONSTANT: {
                XbcFloatConstant m = _factory.createXbcFloatConstant();
                m.setContent(xobj.getFloatString());
                xmobj = m;
            }
            break;
        case LONGLONG_CONSTANT: {
                XbcLonglongConstant m = _factory.createXbcLonglongConstant();
                m.setContent(
                    "0x" + Integer.toHexString((int)xobj.getLongHigh()) + " " +
                    "0x" + Integer.toHexString((int)xobj.getLongLow()));
                xmobj = m;
            }
            break;
        case MOE_CONSTANT: {
                XbcMoeConstant m = _factory.createXbcMoeConstant();
                m.setType(getTypeId(xobj));
                m.setContent(xobj.getString());
                xmobj = m;
            }
            break;
            
        // definition and declaration
        
        case FUNCTION_DEFINITION: {
                // (CODE name symbols params body gccAttrs)
                XobjList identList = (XobjList)xobj.getArg(1);
                XobjList paramList = (XobjList)xobj.getArg(2);
                XobjList bodyList = (XobjList)xobj.getArgOrNull(3);
                Xobject gccAttrs = xobj.getArgOrNull(4);
            
                XbcFunctionDefinition m = _factory.createXbcFunctionDefinition();
                m.setName(transName(xobj.getArg(0)));
                m.setSymbols(transSymbols(identList));
                m.setParams(transFuncDefParams(paramList, identList));
                m.setBody(transBody(bodyList));
                m.setGccAttributes((XbcGccAttributes)trans(gccAttrs));
                xmobj = m;
            }
            break;
        case VAR_DECL: {
                // (CODE name init gccAsm)
                XbcVarDecl m = _factory.createXbcVarDecl();
                m.setName(transName(xobj.getArg(0)));
                m.setValue(transValue(xobj.getArgOrNull(1)));
                m.setGccAsm((XbcGccAsm)trans(xobj.getArgOrNull(2)));
                xmobj = m;
            }
            break;
        case FUNCTION_DECL: {
                // (CODE name () gccAsm)
                XbcFunctionDecl m = _factory.createXbcFunctionDecl();
                m.setName(transName(xobj.getArg(0)));
                m.setGccAsm((XbcGccAsm)trans(xobj.getArgOrNull(2)));
                xmobj = m;
            }
            break;

        // statements

        case COMPOUND_STATEMENT: {
                XobjList identList = (XobjList)xobj.getArg(0);
                XobjList declList = (XobjList)xobj.getArg(1);
                XobjList addDeclList = getDeclForNotDeclared(identList);
                if(addDeclList != null) {
                    if(declList == null)
                        declList = Xcons.List();
                    addDeclList.reverse();
                    for(Xobject a : addDeclList)
                        declList.insert(a);
                }

                XbcCompoundStatement m = _factory.createXbcCompoundStatement();
                m.setSymbols(transSymbols(identList));
                m.setDeclarations(transDeclarations(declList));
                m.setBody(transBody(xobj.getArg(2)));
                xmobj = m;
            }
            break;
        case EXPR_STATEMENT: {
                if (xobj.getArg(0).Opcode() == Xcode.GCC_ASM_STATEMENT) {
                    return trans(xobj.getArg(0));
                }
                else {
                    XbcExprStatement m = _factory.createXbcExprStatement();
                    m.setExpressions(transExpr(xobj.getArg(0)));
                    xmobj = m;
                }
            }
            break;
        case IF_STATEMENT: {
                XbcIfStatement m = _factory.createXbcIfStatement();
                m.setCondition(transCondition(xobj.getArg(0)));
                XbcThen mthen = _factory.createXbcThen();
                mthen.setStatements((IXbcStatementsChoice)trans(xobj.getArg(1)));
                m.setThen(mthen);
                Xobject xelse = xobj.getArgOrNull(2);
                if(xelse != null) {
                    XbcElse melse = _factory.createXbcElse();
                    melse.setStatements((IXbcStatementsChoice)trans(xobj.getArg(2)));
                    m.setElse(melse);
                }
                xmobj = m;
            }
            break;
        case WHILE_STATEMENT: {
                XbcWhileStatement m = _factory.createXbcWhileStatement();
                m.setCondition(transCondition(xobj.getArg(0)));
                m.setBody(transBody(xobj.getArg(1)));
                xmobj = m;
            }
            break;
        case DO_STATEMENT: {
                XbcDoStatement m = _factory.createXbcDoStatement();
                m.setBody(transBody(xobj.getArg(0)));
                m.setCondition(transCondition(xobj.getArg(1)));
                xmobj = m;
            }
            break;
        case FOR_STATEMENT: {
                XbcForStatement m = _factory.createXbcForStatement();
                XbcInit minit = _factory.createXbcInit();
                XbcCondition mcond = _factory.createXbcCondition();
                XbcIter miter = _factory.createXbcIter();
                minit.setExpressions(transExpr(xobj.getArg(0)));
                mcond.setExpressions(transExpr(xobj.getArg(1)));
                miter.setExpressions(transExpr(xobj.getArg(2)));
                m.setInit(minit);
                m.setCondition(mcond);
                m.setIter(miter);
                m.setBody(transBody(xobj.getArg(3)));
                xmobj = m;
            }
            break;
        case SWITCH_STATEMENT: {
                XbcSwitchStatement m = _factory.createXbcSwitchStatement();
                m.setValue(transValue(xobj.getArg(0)));
                m.setBody(transBody(xobj.getArg(1)));
                xmobj = m;
            }
            break;
        case BREAK_STATEMENT: {
                xmobj = _factory.createXbcBreakStatement();
            }
            break;
        case CONTINUE_STATEMENT: {
                xmobj = _factory.createXbcContinueStatement();
            }
            break;
        case RETURN_STATEMENT: {
                XbcReturnStatement m = _factory.createXbcReturnStatement();
                m.setExpressions(transExpr(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case GOTO_STATEMENT: {
                XbcGotoStatement m = _factory.createXbcGotoStatement();
                Xobject x = xobj.getArg(0);
                if(x.Opcode() == Xcode.IDENT) {
                    m.setContent(transName(x));
                } else {
                    m.setContent((IXbcGotoStatementChoice)transExpr(x));
                }
                xmobj = m;
            }
            break;
        case STATEMENT_LABEL: {
                XbcStatementLabel m = _factory.createXbcStatementLabel();
                m.setName(transName(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case CASE_LABEL: {
                XbcCaseLabel m = _factory.createXbcCaseLabel();
                m.setValue(transValue(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case DEFAULT_LABEL: {
                xmobj = _factory.createXbcDefaultLabel();
            }
            break;
            
        // expression
            
        case CONDITIONAL_EXPR: {
                XbcCondExpr m = _factory.createXbcCondExpr();
                m.setExpressions1(transExprOrError(xobj.getArg(0)));
                Xobject a = xobj.getArg(1);
                m.setExpressions2(transExprOrError(a.getArg(0)));
                m.setExpressions3(transExprOrError(a.getArg(1)));
                xmobj = m;
            }
            break;
        case COMMA_EXPR: {
                XbcCommaExpr m = _factory.createXbcCommaExpr();
                for(Xobject a : (XobjList)xobj) {
                    m.addExpressions(transExprOrError(a));
                }
                xmobj = m;
            }
            break;
        case DESIGNATED_VALUE: {
                XbcDesignatedValue m = _factory.createXbcDesignatedValue();
                m.setMember(xobj.getArg(0).getName());
                m.setContent((IXbcDesignatedValueChoice)transExprOrValue(xobj.getArg(1)));
                xmobj = m;
            }
            break;
        case COMPOUND_VALUE: {
                XbcCompoundValueExpr m = _factory.createXbcCompoundValueExpr();
                m.setValue(transValue(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case COMPOUND_VALUE_ADDR: {
                XbcCompoundValueAddrExpr m = _factory.createXbcCompoundValueAddrExpr();
                m.setValue(transValue(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case ADDR_OF: {
                IXbcSymbolAddr m = null;
                IXbcMember mm = null;
                Xobject operand = xobj.operand();
                switch(operand.Opcode()) {
                case VAR:
                case VAR_ADDR:
                    m = _factory.createXbcVarAddr();
                    break;
                //case ARRAY_REF:
                case ARRAY_ADDR: /* illegal but convert */
                    m = _factory.createXbcArrayAddr();
                    break;
                case FUNC_ADDR:
                    m = _factory.createXbcFuncAddr();
                    break;
                case MEMBER_REF:
                case MEMBER_ADDR:
                    mm = _factory.createXbcMemberAddr();
                    break;
                case MEMBER_ARRAY_REF:
                case MEMBER_ARRAY_ADDR:
                    mm = _factory.createXbcMemberArrayAddr();
                    break;
                case POINTER_REF:
                    // reduce (ADDR_OF (POINTER_REF expr)) => expr
                    return transOrError(operand.getArg(0));
                default:
                    fatal("cannot apply ADDR_OF to " + operand.toString());
                }
                
                if(m != null) {
                    m.setContent(operand.getName());
                    xmobj = (XmObj)m;
                } else {
                    mm.setExpressions(transExpr(operand.getArg(0)));
                    mm.setMember(operand.getArg(1).getName());
                    xmobj = (XmObj)mm;
                }
            }
            break;
        case FUNCTION_CALL: {
                XbcFunctionCall m = _factory.createXbcFunctionCall();
                XbcFunction mfunc = _factory.createXbcFunction();
                XbcArguments margs = _factory.createXbcArguments();
                mfunc.setExpressions(transExprOrError(xobj.getArg(0)));
                XobjList params = (XobjList)xobj.getArg(1);
                if(params != null) {
                    for(Xobject a : params) {
                        margs.addExpressions(transExprOrError(a));
                    }
                }
                m.setFunction(mfunc);
                m.setArguments(margs);
                xmobj = m;
            }
            break;
        case SIZE_OF_EXPR: {
                xmobj = (XmObj)transSizeOrAlignOf(xobj);
            }
            break;
        case CAST_EXPR: {
                XbcCastExpr m = _factory.createXbcCastExpr();
                m.setContent((IXbcCastExprChoice)transOrError(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case TYPE_NAME: {
                XbcTypeName m = _factory.createXbcTypeName();
                m.setType(xobj.Type().getXcodeCId());
                xmobj = m;
            }
            break;
        case BUILTIN_OP: {
                XbcBuiltinOp m = _factory.createXbcBuiltinOp();
                m.setName(xobj.getArg(0).getName());
                m.setIsId(intFlagToBoolStr(xobj.getArg(1)));
                m.setIsAddrOf(intFlagToBoolStr(xobj.getArg(2)));
                XobjList args = (XobjList)xobj.getArgOrNull(3);
                if(args != null) {
                    for(Xobject a : args)
                        m.addContent((IXbcBuiltinOpChoice)transOrError(a));
                }
                xmobj = m;
            }
            break;

        // expression IXbcUnaryExpr

        case POINTER_REF:
            // reduce (POINTER_REF (ADDR_OF expr)) => expr
            if(xobj.getArg(0).Opcode() == Xcode.ADDR_OF)
                return transOrError(xobj.getArg(0).getArg(0));
            // no break
        case UNARY_MINUS_EXPR:
        case LOG_NOT_EXPR:
        case BIT_NOT_EXPR:
        case PRE_INCR_EXPR:
        case PRE_DECR_EXPR:
        case POST_INCR_EXPR:
        case POST_DECR_EXPR: {
                IXbcUnaryExpr m = (IXbcUnaryExpr)newXmObj(xobj);
                m.setExpressions((IXbcExpressionsChoice)transOrError(xobj.getArg(0)));
                xmobj = (XmObj)m;
            }
            break;
        
        // expression IXbcBinaryExpr
        
        case ASSIGN_EXPR:
        case PLUS_EXPR:
        case ASG_PLUS_EXPR:
        case MINUS_EXPR:
        case ASG_MINUS_EXPR:
        case MUL_EXPR:
        case ASG_MUL_EXPR:
        case DIV_EXPR:
        case ASG_DIV_EXPR:
        case MOD_EXPR:
        case ASG_MOD_EXPR:
        case LSHIFT_EXPR:
        case ASG_LSHIFT_EXPR:
        case RSHIFT_EXPR:
        case ASG_RSHIFT_EXPR:
        case BIT_AND_EXPR:
        case ASG_BIT_AND_EXPR:
        case BIT_OR_EXPR:
        case ASG_BIT_OR_EXPR:
        case BIT_XOR_EXPR:
        case ASG_BIT_XOR_EXPR:
        case LOG_EQ_EXPR:
        case LOG_NEQ_EXPR:
        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case LOG_AND_EXPR:
        case LOG_OR_EXPR: {
                IXbcBinaryExpr m = (IXbcBinaryExpr)newXmObj(xobj);
                m.setExpressions1((IXbcExpressionsChoice)transOrError(xobj.getArg(0)));
                m.setExpressions2((IXbcExpressionsChoice)transOrError(xobj.getArg(1)));
                xmobj = (XmObj)m;
            }
            break;
            
        // symbol reference

        case VAR:
        case VAR_ADDR:
        // case ARRAY_REF:
        case ARRAY_ADDR: {
                xmobj = (XmObj)transVarRef(xobj);
            }
            break;
        case FUNC_ADDR: {
                XbcFuncAddr m = _factory.createXbcFuncAddr();
                m.setContent(xobj.getName());
                xmobj = m;
            }
            break;
        case MEMBER_REF:
        case MEMBER_ADDR:
        case MEMBER_ARRAY_REF:
        case MEMBER_ARRAY_ADDR: {
                xmobj = (XmObj)transMember(xobj);
            }
            break;

        // gcc syntax

        case GCC_ATTRIBUTES: {
                XbcGccAttributes m = _factory.createXbcGccAttributes();
                for(Xobject a : (XobjList)xobj) {
                    m.addGccAttribute((XbcGccAttribute)trans(a));
                }
                xmobj = m;
            }
            break;
        case GCC_ATTRIBUTE: {
                XbcGccAttribute m = _factory.createXbcGccAttribute();
                m.setName(xobj.getArg(0).getString());
                for(Xobject a : (XobjList)xobj.getArg(1)) {
                    m.addExpressions(transExprOrError(a));
                }
                xmobj = m;
            }
            break;
        case GCC_ASM: {
                XbcGccAsm m = _factory.createXbcGccAsm();
                m.setStringConstant((XbcStringConstant)trans(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case GCC_ASM_DEFINITION: {
                XbcGccAsmDefinition m = _factory.createXbcGccAsmDefinition();
                m.setStringConstant((XbcStringConstant)trans(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case GCC_ASM_STATEMENT: {
                // (CODE is_volatile string_constant operand1 operand2 clobbers)
                XbcGccAsmStatement m = _factory.createXbcGccAsmStatement();
                m.setIsVolatile(intFlagToBoolStr(xobj.getArg(0)));
                m.setStringConstant((XbcStringConstant)transExpr(xobj.getArg(1)));
                m.setGccAsmOperands1((XbcGccAsmOperands)trans(xobj.getArgOrNull(2)));
                m.setGccAsmOperands2((XbcGccAsmOperands)trans(xobj.getArgOrNull(3)));
                m.setGccAsmClobbers((XbcGccAsmClobbers)trans(xobj.getArgOrNull(4)));
                xmobj = m;
            }
            break;
        case GCC_ASM_OPERANDS: {
                XbcGccAsmOperands m = _factory.createXbcGccAsmOperands();
                for(Xobject a : (XobjList)xobj) {
                    m.addGccAsmOperand((XbcGccAsmOperand)trans(a));
                }
                xmobj = m;
            }
            break;
        case GCC_ASM_OPERAND: {
                XbcGccAsmOperand m = _factory.createXbcGccAsmOperand();
                m.setExpressions(transExpr(xobj.getArg(0)));

                if (xobj.getArg(1) != null) {
                    m.setMatch(xobj.getArg(1).getString());
                }

                if (xobj.getArg(2) != null) {
                    m.setConstraint(xobj.getArg(2).getString());
                }

                xmobj = m;
            }
            break;
        case GCC_ASM_CLOBBERS: {
                XbcGccAsmClobbers m = _factory.createXbcGccAsmClobbers();
                m.setStringConstant((XbcStringConstant)trans(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case GCC_ALIGN_OF_EXPR: {
                xmobj = (XmObj)transSizeOrAlignOf(xobj);
            }
            break;
        case GCC_MEMBER_DESIGNATOR: {
                XbcGccMemberDesignator m = _factory.createXbcGccMemberDesignator();
                m.setRef(xobj.getArg(0).Type().getXcodeCId());
                m.setMember(xobj.getArg(1).getName());
                m.setExpressions(transExpr(xobj.getArg(2)));
                m.setGccMemberDesignator((XbcGccMemberDesignator)trans(xobj.getArg(3)));
                xmobj = m;
            }
            break;
        case GCC_LABEL_ADDR: {
                XbcGccLabelAddr m = _factory.createXbcGccLabelAddr();
                m.setContent(xobj.getName());
                xmobj = m;
            }
            break;
        case GCC_COMPOUND_EXPR: {
                XbcGccCompoundExpr m = _factory.createXbcGccCompoundExpr();
                m.setCompoundStatement((XbcCompoundStatement)trans(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case GCC_RANGED_CASE_LABEL: {
                XbcGccRangedCaseLabel m = _factory.createXbcGccRangedCaseLabel();
                m.setValue1(transValue(xobj.getArg(0)));
                m.setValue2(transValue(xobj.getArg(0)));
                xmobj = m;
            }
            break;
            
        // xmp syntax
        case ARRAY_REF: {
                XbcArrayRef m = _factory.createXbcArrayRef();
                m.setArrayAddr((XbcArrayAddr)trans(xobj.getArg(0)));

                XobjList exprList = (XobjList)xobj.getArg(1);
                int exprListSize = exprList.Nargs();
                IXbcExpressionsChoice[] exprs = new IXbcExpressionsChoice[exprListSize];
                for (int i = 0; i < exprListSize; i++) {
                  exprs[i] = transExpr(exprList.getArg(i));
                }

                m.setExpressions(exprs);
                xmobj = m;
            }
            break;
        case SUB_ARRAY_REF: {
                XbcSubArrayRef m = _factory.createXbcSubArrayRef();
                m.setArrayAddr((XbcArrayAddr)trans(xobj.getArg(0)));

                XobjList exprList = (XobjList)xobj.getArg(1);
                int exprListSize = exprList.Nargs();
                IXbcSubArrayDimensionChoice[] exprs = new IXbcSubArrayDimensionChoice[exprListSize];
                for (int i = 0; i < exprListSize; i++) {
                  exprs[i] = (IXbcSubArrayDimensionChoice)trans(exprList.getArg(i));
                }

                m.setSubArrayDimension(exprs);
                xmobj = m;
            }
            break;
        case LOWER_BOUND: {
                XbcLowerBound m = _factory.createXbcLowerBound();
                m.setExpressions(transExpr(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case UPPER_BOUND: {
                XbcUpperBound m = _factory.createXbcUpperBound();
                m.setExpressions(transExpr(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case STEP: {
                XbcStep m = _factory.createXbcStep();
                m.setExpressions(transExpr(xobj.getArg(0)));
                xmobj = m;
            }
            break;
        case CO_ARRAY_REF: {
                XbcCoArrayRef m = _factory.createXbcCoArrayRef();
                //m.setName(transName(xobj.getArg(0)));
		m.setContent((IXbcCoArrayRefChoice1)transExpr(xobj.getArg(0)));
                for(Xobject a : (XobjList)xobj.getArg(1)) {
                    m.addExpressions(transExpr(a));
                }
                if(xobj.getScope() != null)
                    m.setScope(xobj.getScope().toXcodeString());
                xmobj = m;
            }
            break;
        case CO_ARRAY_ASSIGN_EXPR: {
                XbcCoArrayAssignExpr m = _factory.createXbcCoArrayAssignExpr();
                m.setExpressions1(transExpr(xobj.getArg(0)));
                m.setExpressions1(transExpr(xobj.getArg(1)));
                xmobj = m;
            }
            break;

        // directive

        case PRAGMA_LINE: {
                XbcPragma m = _factory.createXbcPragma();
                m.setContent(xobj.getArg(0).getString());
                xmobj = m;
            }
            break;
        case TEXT: {
                XbcText m = _factory.createXbcText();
                m.setContent(xobj.getArg(0).getString());
                xmobj = m;
            }
            break;
        case OMP_PRAGMA: 
        case XMP_PRAGMA: {
                XbcText m = _factory.createXbcText();
                m.setContent("/* ignored Xcode." + xobj.Opcode().toXcodeString() + " */");
                xmobj = m;
            }
            break;
        case INDEX_RANGE: {
                XbcIndexRange m = _factory.createXbcIndexRange();
                m.setLowerBound((XbcLowerBound)trans(xobj.getArg(0)));
                m.setUpperBound((XbcUpperBound)trans(xobj.getArg(1)));
                m.setStep((XbcStep)trans(xobj.getArg(2)));
                xmobj = m;
            }
            break;
        default: {
                fatal_dump("cannot convert Xcode to XcodeML.", xobj);
            }
        }

        if(xmobj instanceof IXbcTypedExpr &&
            (!(xmobj instanceof XbcBuiltinOp) || (xobj.Type() != null))) {
            IXbcTypedExpr i = (IXbcTypedExpr)xmobj;
            i.setType(getTypeId(xobj));
        }
        
        if(xmobj instanceof IXbLineNo) {
            IXbLineNo i = (IXbLineNo)xmobj;
            if(xobj.getLineNo() != null) {
                i.setLineno("" + xobj.getLineNo().lineNo());
                i.setFile(xobj.getLineNo().fileName());
            }
        }
        
        if(xmobj instanceof IXbcAnnotation) {
            IXbcAnnotation i = (IXbcAnnotation)xmobj;
            if(xobj.isGccSyntax())
                i.setIsGccSyntax(TRUE_STR);
            if(xobj.isSyntaxModified())
                i.setIsModified(TRUE_STR);
        }
        
        if(xmobj instanceof IXbcHasGccExtension) {
            IXbcHasGccExtension i = (IXbcHasGccExtension)xmobj;
            if(xobj.isGccExtension())
                i.setIsGccExtension(TRUE_STR);
        }
       
        return xmobj;
    }
    
    private XbcTypeTable transTypeTable(List<Xtype> xtypeList)
    {
        XbcTypeTable xmobj = _factory.createXbcTypeTable();
        for(Xtype xtype : xtypeList) {
            xmobj.addTypes(transType(xtype));
        }
        return xmobj;
    }
    
    private String toBoolStr(boolean enabled)
    {
        return (enabled) ? TRUE_STR : null;
    }
    
    private String intFlagToBoolStr(Xobject flag)
    {
        return (flag.getInt() == 1) ? TRUE_STR : null;
    }
    
    private IXbcTypesChoice transType(Xtype type)
    {
        IXbcType xmtype = null;
        
        if(type.copied != null) {
            XbcBasicType t = _factory.createXbcBasicType();
            t.setName(type.copied.getXcodeCId());
            xmtype = t;
        } else {
            switch(type.getKind()) {
            case Xtype.BASIC: {
                    XbcBasicType t = _factory.createXbcBasicType();
                    t.setName(BasicType.getTypeInfo(type.getBasicType()).cname);
                    xmtype = t;
                }
                break;
                
            case Xtype.ENUM: {
                    if(type.copied != null) {
                        XbcBasicType t = _factory.createXbcBasicType();
                        t.setName(type.copied.getXcodeCId());
                        xmtype = t;
                    } else {
                        XbcEnumType t = _factory.createXbcEnumType();
                        t.setSymbols(transSymbols(type.getMemberList()));
                        xmtype = t;
                    }
                }
                break;
                
            case Xtype.STRUCT: {
                    if(type.copied != null) {
                        XbcBasicType t = _factory.createXbcBasicType();
                        t.setName(type.copied.getXcodeCId());
                        xmtype = t;
                    } else {
                        XbcStructType t = _factory.createXbcStructType();
                        t.setSymbols(transSymbols(type.getMemberList()));
                        xmtype = t;
                    }
                }
                break;
                
            case Xtype.UNION: {
                    if(type.copied != null) {
                        XbcBasicType t = _factory.createXbcBasicType();
                        t.setName(type.copied.getXcodeCId());
                        xmtype = t;
                    } else {
                        XbcUnionType t = _factory.createXbcUnionType();
                        t.setSymbols(transSymbols(type.getMemberList()));
                        xmtype = t;
                    }
                }
                break;
                
            case Xtype.ARRAY: {
                    XbcArrayType t = _factory.createXbcArrayType();
                    t.setElementType(type.getRef().getXcodeCId());
                    if(type.getArraySize() >= 0)
                        t.setArraySize1("" + type.getArraySize());
                    else if(type.getArraySizeExpr() != null) {
                        t.setArraySize1("*");
                        t.setArraySize2(transArraySize(type.getArraySizeExpr()));
                    }
                    t.setIsStatic(toBoolStr(type.isArrayStatic()));
                    xmtype = t;
                }
                break;
                
            case Xtype.FUNCTION: {
                    XbcFunctionType t = _factory.createXbcFunctionType();
                    t.setReturnType(type.getRef().getXcodeCId());
                    t.setParams(transFuncTypeParams((XobjList)type.getFuncParam()));
                    t.setIsStatic(toBoolStr(type.isFuncStatic()));
                    t.setIsInline(toBoolStr(type.isInline()));
                    t.setGccAttributes((XbcGccAttributes)trans(type.getGccAttributes()));
                    xmtype = t;
                }
                break;
            case Xtype.POINTER: {
                    XbcPointerType t = _factory.createXbcPointerType();
                    t.setRef(type.getRef().getXcodeCId());
                    xmtype = t;
                }
                break;
            
            case Xtype.XMP_CO_ARRAY: {
                    XbcCoArrayType t = _factory.createXbcCoArrayType();
                    t.setElementType(type.getRef().getXcodeCId());
                    if(type.getArraySize() >= 0)
                        t.setArraySize1("" + type.getArraySize());
                    else if(type.getArraySizeExpr() != null) {
                        t.setArraySize1("*");
                        t.setArraySize2(transArraySize(type.getArraySizeExpr()));
                    }
                    return (IXbcTypesChoice)xmtype;
                }
                
            default:
                fatal("cannot convert type_kind:" + type.getKind());
            }
        }

        xmtype.setType(type.getXcodeCId());
        xmtype.setIsConst(toBoolStr(type.isConst()));
        xmtype.setIsRestrict(toBoolStr(type.isRestrict()));
        xmtype.setIsVolatile(toBoolStr(type.isVolatile()));
        xmtype.setGccAttributes((XbcGccAttributes)trans(type.getGccAttributes()));
        
        return (IXbcTypesChoice)xmtype;
    }
    
    private XbcGlobalSymbols transGlobalSymbols(Xobject xobj)
    {
        XobjList identList = (XobjList)xobj;
        XbcGlobalSymbols xmobj = _factory.createXbcGlobalSymbols();
        if(identList != null) {
            for(Xobject ident : identList) {
                xmobj.addId(transIdent((Ident)ident));
            }
        }
        return xmobj;
    }
    
    private XbcSymbols transSymbols(Xobject xobj)
    {
        XobjList identList = (XobjList)xobj;
        XbcSymbols xmobj = _factory.createXbcSymbols();
        if(identList != null) {
            for(Xobject ident : identList) {
                xmobj.addContent(transIdent((Ident)ident));
            }
        }
        return xmobj;
    }
    
    private XbcId transIdent(Ident ident)
    {
        if(ident == null)
            return null;
        XbcId xmid = _factory.createXbcId();
        
        // type
        if(ident.Type() != null)
            xmid.setType(ident.Type().getXcodeCId());
        
        // name
        XbcName xmname = _factory.createXbcName();
        xmname.setContent(ident.getName());
        xmid.setName(xmname);
        
        // sclass
        if(ident.getStorageClass() != null)
            xmid.setSclass(ident.getStorageClass().toXcodeString());

        // bit field
        if(ident.getBitField() > 0)
            xmid.setBitField1("" + ident.getBitField());
        else if(ident.getBitFieldExpr() != null) {
            xmid.setBitField1("*");
            xmid.setBitField2(transBitField(ident.getBitFieldExpr()));
        }
        
        // enum member value
        xmid.setValue(transValue(ident.getEnumValue()));
        
        // gcc attributes
        xmid.setGccAttributes((XbcGccAttributes)trans(ident.getGccAttributes()));
        
        return xmid;
    }
    
    private XbcBitField transBitField(Xobject expr)
    {
        if(expr == null)
            return null;
        XbcBitField xmbf = _factory.createXbcBitField();
        xmbf.setExpressions(transExpr(expr));
        return xmbf;
    }
    
    private XbcGlobalDeclarations transGlobalDeclarations(XobjectDefEnv defList)
    {
        XbcGlobalDeclarations xmgdecls = _factory.createXbcGlobalDeclarations();
        for(XobjectDef def : defList) {
            IXbcGlobalDeclarationsChoice xmdef = (IXbcGlobalDeclarationsChoice)transDef(def);
            if(xmdef != null)
                xmgdecls.addContent(xmdef);
        }
        return xmgdecls;
    }
    
    private XbcDeclarations transDeclarations(Xobject xobj)
    {
        XbcDeclarations xmdecls = _factory.createXbcDeclarations();
        if(xobj != null) {
            for(Xobject a : (XobjList)xobj) {
                xmdecls.addContent((IXbcDeclarationsChoice)transOrError(a));
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
        return transOrError(defObj);
    }
    
    private XbcArraySize transArraySize(Xobject expr)
    {
        if(expr == null)
            return null;
        XbcArraySize xmas = _factory.createXbcArraySize();
        xmas.setExpressions(transExpr(expr));
        return xmas;
    }
    
    private IXbcExpressionsChoice transExpr(Xobject expr)
    {
        return (IXbcExpressionsChoice)trans(expr);
    }
    
    private IXbcExpressionsChoice transExprOrError(Xobject expr)
    {
        return (IXbcExpressionsChoice)transOrError(expr);
    }
    
    private XmObj transExprOrValue(Xobject expr)
    {
        if(expr.Opcode() == Xcode.LIST) {
            return transValue(expr);
        }
        return (XmObj)transExpr(expr);
    }
    
    private XbcValue transValue(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbcValue mval = _factory.createXbcValue();
        mval.setContent((IXbcValueChoice)transValueChild(xobj));
        return mval;
    }
    
    private XmObj transValueChild(Xobject xobj)
    {
        if(xobj.Opcode() == Xcode.LIST) {
            XbcCompoundValue mcomp = _factory.createXbcCompoundValue();
            for(Xobject a : (XobjList)xobj) {
                IXbcCompoundLiteralChoice mval = (IXbcCompoundLiteralChoice)transValueChild(a);
                if(mval != null)
                    mcomp.addCompoundLiteral(mval);
            }
            return mcomp;
        } else {
            return trans(xobj);
        }
    }
    
    private XbcParams transFuncTypeParams(XobjList paramList)
    {
        XbcParams xmparams = _factory.createXbcParams();
        if(paramList != null) {
            if(paramList.isEmpty()) {
                XbcName name = new XbcName();
                name.setType(BasicType.getTypeInfo(BasicType.VOID).cname);
                xmparams.addName(name);
            } else {
                for(Xobject param : paramList) {
                    if(param == null)
                        xmparams.setEllipsis(TRUE_STR);
                    else
                        xmparams.addName(transName(param));
                }
            }
        }
        return xmparams;
    }
    
    
    private XbcParams transFuncDefParams(XobjList declList, XobjList identList)
    {
        XbcParams mparams = new XbcParams();
        
        if(declList != null) {
            for(Xobject a: declList) {
                if(a == null) {
                    mparams.setEllipsis(TRUE_STR);
                    break;
                }
                if(a.Opcode() != Xcode.VAR_DECL)
                    fatal("not VAR_DECL : " + a.toString());
                String paramName = a.getArg(0).getName();
                Xtype type = null;
                for(Xobject i : identList) {
                    if(i.getName().equals(paramName)) {
                        type = i.Type();
                        break;
                    }
                }
                XbcName mname = _factory.createXbcName();
                mname.setContent(paramName);
                mname.setType(type.getXcodeCId());
                mparams.addName(mname);
            }
        }

        if(identList != null) {
            for(Xobject a : identList) {
                Ident id = (Ident)a;
                if(id.isDeclared() || id.getStorageClass() != StorageClass.PARAM)
                    continue;
                XbcName mname = _factory.createXbcName();
                mname.setContent(id.getName());
                mname.setType(id.Type().getXcodeCId());
                mparams.addName(mname);
            }
        }
        
        return mparams;
    }
    
    private XbcName transName(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbcName xmname = _factory.createXbcName();
        if(xobj.Type() != null)
            xmname.setType(xobj.Type().getXcodeCId());
        xmname.setContent(xobj.getString());
        return xmname;
    }
    
    private XbcBody transBody(Xobject xobj)
    {
        XbcBody xmbody = _factory.createXbcBody();
        
        if(xobj != null) {
            if(xobj.Opcode() == Xcode.LIST) {
                for(Xobject a : (XobjList)xobj) {
                    IXbcStatementsChoice xmstmt = (IXbcStatementsChoice)trans(a);
                    if(xmstmt != null)
                        xmbody.addStatements(xmstmt);
                }
            } else {
                IXbcStatementsChoice xmstmt = (IXbcStatementsChoice)trans(xobj);
                if(xmstmt != null)
                    xmbody.addStatements(xmstmt);
            }
        }
        return xmbody;
    }
    
    private XbcCondition transCondition(Xobject xobj)
    {
        if(xobj == null)
            return null;
        XbcCondition xmcond = _factory.createXbcCondition();
        xmcond.setExpressions(transExpr(xobj));
        return xmcond;
    }
    
    private IXbcVarRef transVarRef(Xobject xobj)
    {
        if(xobj == null)
            return null;
        IXbcVarRef xmvr = (IXbcVarRef)newXmObj(xobj);
        if(xobj.getScope() != null)
            xmvr.setScope(xobj.getScope().toXcodeString());
        xmvr.setContent(xobj.getName());
        return xmvr;
    }
    
    private IXbcMember transMember(Xobject xobj)
    {
        IXbcMember xmmem = (IXbcMember)newXmObj(xobj);
        xmmem.setExpressions(transExpr(xobj.getArg(0)));
        xmmem.setMember(xobj.getArg(1).getString());
        return xmmem;
    }
    
    private IXbcSizeOrAlignExpr transSizeOrAlignOf(Xobject xobj)
    {
        IXbcSizeOrAlignExpr xsa = (IXbcSizeOrAlignExpr)newXmObj(xobj);
        xsa.setExprOrType((IXbcExprOrTypeChoice)transOrError(xobj.getArg(0)));
        return xsa;
    }
    
    private List<XbcText> transLines(List<String> lines)
    {
        ArrayList<XbcText> textList = new ArrayList<XbcText>(lines != null ? lines.size() : 0);
        
        if(lines == null)
            return textList;
        
        for(String line : lines) {
            if(line == null)
                continue;
            String[] splines = line.split("\n");
            for(String spline : splines) {
                XbcText text = new XbcText();
                text.setContent(spline);
                textList.add(text);
            }
        }
        
        return textList;
    }
}
