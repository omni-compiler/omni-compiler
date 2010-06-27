/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import java.util.List;
import java.util.LinkedList;
import java.util.Queue;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcArrayType;
import xcodeml.c.type.XcBaseTypeEnum;
import xcodeml.c.type.XcIntegerType;
import xcodeml.c.type.XcStructType;
import xcodeml.c.type.XcTypeEnum;
import xcodeml.c.type.XcVoidType;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcPointerType;
import xcodeml.c.type.XcType;
import xcodeml.c.type.XcXmpCoArrayType;

/**
 * An factory creates C language object from the object which represents XcalableMP extension.
 */
public final class XcXmpFactory
{
    private static int _tmpVarId = 0;

    private static final String _tmpValue = "__xmp_tmp_value_";

    private static final String _mainFunctionName = "xmp_main";

    private static final XcIdent _xmpCoAPutFuncAddr  = new XcIdent("__xmp_coarray_put");

    private static final XcIdent _xmpCoAGetFuncAddr  = new XcIdent("__xmp_coarray_get");

    private static final XcIdent _xmpSuAAsgFuncAddr  = new XcIdent("__xmp_subarray_asg");

    private static final XcStructType _xmpSubArrayType = new XcStructType("SXMP0", "__xmp_subarray_t");

    private static final XcStructType _xmpSubArrayDimType = new XcStructType("SXMP1", "__xmp_subarray_dim_t");

    private static final XcStructType _xmpCoArrayDimType = new XcStructType("SXMP2", "__xmp_coarray_dim_t");

    private static final XcVoidType _void = new XcVoidType("XMPB0");

    private static final XcPointerType _pvoid = new XcPointerType("XMPP", _void);

    private static final XcConstObj.IntConst _zero =  new XcConstObj.IntConst(0, XcBaseTypeEnum.INT);

    private static final XcConstObj.IntConst _one = new XcConstObj.IntConst(1, XcBaseTypeEnum.INT);

    private static final XcConstObj.IntConst _false = _zero;

    private static final XcConstObj.IntConst _true = _one;

    private static final XcDirectiveObj _includeDirective = new XcDirectiveObj("#include <omxmp_internal.h>");

    /**
     * Creates string be used as a name of temporary variable.
     *
     * @return the name of temporary variable name.
     */
    private static String _createTmpValueName()
    {
        return _tmpValue + _tmpVarId++;
    }

    /**
     * Creates an include directive which links to XcalableMP extensions implements.
     *
     * @return the include directive object. 
     */
    public static XcDirectiveObj createXmpIncludeDirective()
    {
        return _includeDirective;
    }

    /**
     * Renames function which must be main function.
     *
     * @param mainFunc function definition will be renamed.
     */
    public static void renameMain(XcFuncDefObj mainFunc)
    {
        if(mainFunc != null)
            mainFunc.setSymbol(_mainFunctionName);
    }

    private static XcSizeOfExprObj _createSizeOf(XcType type)
    {
        return new XcSizeOfExprObj(XcOperatorEnum.SIZEOF, type);
    }

    /**
     * Creates __xmp_coarray_dim_t array.
     */
    private static XcExprObj _createCoADim(XcType type, List<XcExprObj> dimList)
        throws XmException
    {
        if(type.getTypeEnum() != XcTypeEnum.COARRAY) {
            throw new XmException("type is not coarray type");
        }

        if(dimList.size() == 0) {
            return _zero;
        }

        XcXmpCoArrayType ct = (XcXmpCoArrayType)type;

        XcArrayType at = new XcArrayType("AXMP0");
          at.setArraySize(dimList.size());
          at.setRefType(_xmpCoArrayDimType);

        XcCompoundValueObj cv = new XcCompoundValueObj();
        for(XcExprObj expr : dimList) {

            XcCompoundValueObj inner_cv = new XcCompoundValueObj();
            cv.addChild(inner_cv);

            inner_cv.addChild(expr);

            XcExprObj arraySizeExpr = ct.getArraySizeAsExpr();
            if(arraySizeExpr != null) {
                inner_cv.addChild(arraySizeExpr);
            } else {
                inner_cv.addChild(_zero);
            }

            type = ct.getRefType();

            if(type.getTypeEnum() != XcTypeEnum.COARRAY)
                break;

            ct = (XcXmpCoArrayType)type;
        }

        XcCompoundValueObj.Ref compoundRef = new XcCompoundValueObj.Ref(at, cv);

        return compoundRef;
    }

    /**
     * Creates an address reference expression. 
     * 
     * @param expr if expr is variable reference expression, it will changed to an address reference expression.
     * @return an address reference expression.
     */
    private static XcExprObj _getRefAddr(XcExprObj expr)
    {
        if (expr instanceof XcXmpCoArrayRefObj){
            ((XcXmpCoArrayRefObj)expr).setNeedAddr();

        } else if (expr instanceof XcCompoundValueObj.Ref) {
            XcType type = ((XcCompoundValueObj.Ref)expr).getType();
            XcExprObj value = ((XcCompoundValueObj.Ref)expr).getValue();

            expr = new XcCompoundValueObj.AddrRef(type, value);

        } else if(expr instanceof XcRefObj) {
            XcRefObj refObj = (XcRefObj)expr;

            switch(refObj.getRefEnum()) {
            case POINTER_REF:
                expr = refObj.getExpr();
                break;
            case MEMBER_REF:
                XcExprObj content = refObj.getExpr();
                XcExprObj addr = _getRefAddr(content);

                expr = new XcRefObj.MemberAddr(addr, ((XcRefObj.MemberRef)refObj).getMember());
                break;
            default:
                break;
            }
        } else if (((expr instanceof XcIdent) &&
                    ((XcIdent)expr).getType().getTypeEnum() == XcTypeEnum.ARRAY) == false) {
            expr = new XcRefObj.Addr(expr);
        }

        return expr;
    }

    /**
     * Creates __xmp_coarray_get function call.
     *
     * @param coArrayRef object.
     * @return __xmp_coarray_get function call.
     * @throws XmException if coArrayRef has incomplete type.
     */
    public static XcExprObj createCoAGetFuncCall(XcXmpCoArrayRefObj refObj)
        throws XmException
    {
        if(refObj.getTempVar() == null)
            return refObj.getContent();

        XcExprObj srcExpr = refObj.getContent();
        srcExpr = _getRefAddr(srcExpr);

        XcCastObj srcAddr = new XcCastObj(_pvoid, srcExpr);
        XcCastObj dstAddr = new XcCastObj(_pvoid, new XcRefObj.Addr(refObj.getTempVar()));

        XcXmpCoArrayType ct = (XcXmpCoArrayType)refObj.getType();
        XcExprObj sizeOfObj = _createSizeOf(refObj.getElementType());

        XcFuncCallObj funcCall = new XcFuncCallObj();
          funcCall.setAddrExpr(_xmpCoAGetFuncAddr);
          funcCall.addChild(dstAddr);
          funcCall.addChild(_createCoADim(ct, refObj.getDimList()));
          funcCall.addChild(new XcConstObj.IntConst(refObj.getDimList().size(), XcBaseTypeEnum.INT));
          funcCall.addChild(srcAddr);
          funcCall.addChild(sizeOfObj);

        XcOperatorObj commaObj = new XcOperatorObj(XcOperatorEnum.COMMA, new XcExprObj[2]);
        commaObj.addChild(funcCall);

        if(refObj.isNeedAddr()) {
            commaObj.addChild(new XcRefObj.Addr(refObj.getTempVar()));
        } else {
            commaObj.addChild(refObj.getTempVar());
        }

        return commaObj;
    }

    /**
     * Creates addr expression of expresion.
     */
    private static XcExprObj _createPutSrcAddr(XcExprObj src, XcType type)
    {
        if(src instanceof XcConstObj.StringConst) {
            XcConstObj.StringConst strc = (XcConstObj.StringConst)src;
            XcArrayType at = new XcArrayType("XMPARRAY");

            XcConstObj.IntConst strLength =
                new XcConstObj.IntConst(strc.getValue().length() + 1, XcBaseTypeEnum.INT);

            at.setArraySizeExpr(strLength);
            at.setRefType(new XcIntegerType.Char());

            return new XcCompoundValueObj.Ref(at, new XcCompoundValueObj(strc));

        } else if ((src instanceof XcIdent) ||
                   (src instanceof XcRefObj) ||
                   (src instanceof XcXmpCoArrayRefObj)
                   ) {
            return _getRefAddr(src);
        } else {
            XcArrayType at = new XcArrayType("XMPARRAY");
            at.setRefType(type);

            at.setArraySizeExpr(new XcConstObj.IntConst(1, XcBaseTypeEnum.INT));

            XcCompoundValueObj cv = new XcCompoundValueObj(src);

            return new XcCompoundValueObj.AddrRef(at, cv);
        }
    }

    /**
     * Creates __xmp_coarray_put function call.
     *
     * @param assginObj represents assignment to coarray.
     * @return __xmp_coarray_put_function call.
     * @throws XmException thrown if coArrayRef has incomplete type.
     */
    public static XcFuncCallObj createCoAPutFuncCall(XcXmpCoArrayAssignObj assignObj)
        throws XmException
    {
        XcExprObj dst = assignObj.getDst();
        XcExprObj src = assignObj.getSrc();

        XcXmpCoArrayRefObj coArrayDst = _getCoArrayRef(dst);

        if(coArrayDst == null) {
            throw new XmException("coArrayAssign destination tag is not coArrayRef.");
        }

        dst = coArrayDst.getContent();

        XcXmpCoArrayType type = (XcXmpCoArrayType)coArrayDst.getType();

        src = _createPutSrcAddr(src, coArrayDst.getElementType());
        XcCastObj dstAddr = new XcCastObj(_pvoid, _getRefAddr(dst));
        XcCastObj srcAddr = new XcCastObj(_pvoid, src);

        XcSizeOfExprObj sizeOfObj = _createSizeOf(coArrayDst.getElementType());

        if(src instanceof XcCompoundValueObj.Ref) {
            XcType srcType = ((XcCompoundValueObj.Ref)src).getType();
            if(srcType.getTypeEnum() == XcTypeEnum.ARRAY) {
                sizeOfObj.setTypeName(srcType);
            }
        }

        XcFuncCallObj funcCall = new XcFuncCallObj();
        funcCall.setAddrExpr(_xmpCoAPutFuncAddr);
        funcCall.addChild(_createCoADim(type, coArrayDst.getDimList())); /* coarray dimension array */
        funcCall.addChild(new XcConstObj.IntConst(coArrayDst.getDimList().size(), XcBaseTypeEnum.INT));
        funcCall.addChild(dstAddr);
        funcCall.addChild(srcAddr);
        funcCall.addChild(sizeOfObj);

        return funcCall;
    }

    private static XcXmpCoArrayRefObj _getCoArrayRef(XcExprObj obj)
    {
        if(obj instanceof XcXmpCoArrayRefObj)
            return (XcXmpCoArrayRefObj)obj;

        if(obj == null || obj.getChild() == null)
            return null;

        for(XcNode node : obj.getChild()) {
            if((node instanceof XcExprObj) == false)
                continue;

            XcXmpCoArrayRefObj coArrayRefObj = _getCoArrayRef((XcExprObj)node);

            if(coArrayRefObj != null)
                return coArrayRefObj;
        }

        return null;
    }

    /**
     * Creates compound statement include XcalableMP function from expression statement.
     * 
     * @param exprStmt expression statement include coarray ref tags.
     * @return expression statement with xmp put/get function call translated from coarray ref tags.
     */
    public static XcStmtObj createXmpStmt(XcExprStmtObj exprStmt)
    {
        exprStmt.setIsNeedCoArrayGurad(false);

        XcDeclsObj decls = new XcDeclsObj();
        XcExprObj expr = exprStmt.getExprObj();

        _getCoArrayTmpVar(expr, decls);

        XcCompStmtObj compStmt = new XcCompStmtObj();

        if(decls.isEmpty() == false) {
            compStmt.addChild(decls);
            compStmt.addChild(exprStmt);
            return compStmt;
        } else {
            return exprStmt;
        }
    }

    /**
     * Creates an statement include sub array assignment function call from an expression statement.
     * 
     * @param exprStmt an expression statement.
     * @return an statement include sub array assignment function call.
     * @throws XmException thrown if an expression statement does not include subArrayRef.
     */
    public static XcExprObj createSuAAsgFuncCall(XcExprStmtObj exprStmt)
        throws XmException
    {
        if(exprStmt == null)
            return null;

        XcExprObj expr = exprStmt.getExprObj();

        return createSuAAsgFuncCall(expr);
    }

    /**
     * Creates sub array assignment function from expression.
      * 
     *  @param expr an expression.
     *  @return an statement include sub array assignment function call.
     * @throws XmException thrown if an expression does not include subArrayRef.
      */
    public static XcExprObj createSuAAsgFuncCall(XcExprObj expr)
        throws XmException
    {
        if(expr == null) {
            throw new XmException("null expression.");
        }

        XcExprObj dstExpr = null, srcExpr = null;

        if(expr instanceof XcOperatorObj) {

            XcOperatorObj op = (XcOperatorObj)expr;

            if(op.getOperatorEnum() != XcOperatorEnum.ASSIGN)
                return null;

            XcExprObj[] exprs = op.getExprObjs();

            dstExpr = exprs[0];
            srcExpr = exprs[1];

        } else if(expr instanceof XcXmpCoArrayAssignObj) {
            XcXmpCoArrayAssignObj coaAsg = (XcXmpCoArrayAssignObj)expr;

            dstExpr = coaAsg.getDst();
            srcExpr = coaAsg.getSrc();
        }

        if((dstExpr == null) ||
           (dstExpr instanceof XcXmpSubArrayRefObj) == false) {
            throw new XmException("the destination of sub array assingment is not correct expression.");
        }

        if((srcExpr == null) ||
           (srcExpr instanceof XcXmpSubArrayRefObj) == false) {
            throw new XmException("the destination of sub array assingment is not correct expression.");
        }

        XcFuncCallObj funcCall = new XcFuncCallObj();
        funcCall.setAddrExpr(_xmpSuAAsgFuncAddr);
        funcCall.addChild(_createSuAType(dstExpr));
        funcCall.addChild(_createSuAType(srcExpr));

        return funcCall;
    }

    /**
     * Creates compound value of __xmp_sub_array_t *
     */
    private static XcCompoundValueObj.AddrRef _createSuAType(XcExprObj arg)
        throws XmException
    {
        LinkedList<XcXmpSubArrayRefObj> suaQueue = new LinkedList<XcXmpSubArrayRefObj>();
        XcConstObj.IntConst isCoarray = _false;
        XcExprObj coaDim = null;
        XcConstObj.IntConst coaDimSize = null;
        XcXmpSubArrayRefObj sua = null;
        XcType type = null;

        while(arg instanceof XcXmpSubArrayRefObj) {
            sua = (XcXmpSubArrayRefObj)arg;

            type = sua.getArrayType();

            sua.makeCompleteInfo();
            suaQueue.addFirst(sua);
            arg = sua.getExpr();
        }

        for(XcXmpSubArrayRefObj suaElm : suaQueue) {
            if((type instanceof XcArrayType) == false) {
                throw new XmException("subArrayRef applied to no array typed object.");
            }

            suaElm.setArraySize(((XcArrayType)type).getArraySizeAsExpr());
            type = type.getRealType().getRefType();
            suaElm.setUnitType(type);
        }

        if(arg instanceof XcXmpCoArrayRefObj) {
            XcXmpCoArrayRefObj refObj = (XcXmpCoArrayRefObj)arg;

            XcXmpCoArrayType ct = (XcXmpCoArrayType)refObj.getType();

            isCoarray = _true;
            coaDim = _createCoADim(ct, refObj.getDimList());
            coaDimSize = new XcConstObj.IntConst(refObj.getDimList().size(), XcBaseTypeEnum.INT);
        }

        XcCastObj subArray = new XcCastObj(_pvoid, _getRefAddr(arg));
        XcCompoundValueObj.Ref suaDimList = _createSuADimList(suaQueue);
        XcConstObj.IntConst suaDimListSize = new XcConstObj.IntConst(suaQueue.size(), XcBaseTypeEnum.INT);

        XcCompoundValueObj cv = new XcCompoundValueObj();
        cv.addChild(subArray);
        cv.addChild(suaDimList);
        cv.addChild(suaDimListSize);
        cv.addChild(isCoarray);

        if(coaDim != null) {
            cv.addChild(coaDim);
            cv.addChild(coaDimSize);
        }

        XcCompoundValueObj.AddrRef obj = new XcCompoundValueObj.AddrRef();
        obj.setType(_xmpSubArrayType);
        obj.addChild(cv);

        return obj;
    }

    /**
     * Creates compound value of sub array dimensions.
     *
     * ex) (__xmp_subarray_dim_t[]){{0,3,1,sizeof(int[4])},{0,3,1,sizeof(int)}}
     */
    private static XcCompoundValueObj.Ref _createSuADimList(Queue<XcXmpSubArrayRefObj> suaQueue)
    {
        XcArrayType at = new XcArrayType("AXMP0");
          at.setArraySize(suaQueue.size());
          at.setRefType(_xmpSubArrayDimType);

        XcCompoundValueObj cv = new XcCompoundValueObj();
        for(XcXmpSubArrayRefObj refObj : suaQueue) {
            cv.addChild(_createSubArrayDimensionType(refObj));
        }

        XcCompoundValueObj.Ref obj = new XcCompoundValueObj.Ref();
          obj.setType(at);
          obj.addChild(cv);

        return obj;
    }

    /**
     * Creates an object represents (expr - 1).
     */
    private static XcOperatorObj _createMinusOne(XcExprObj expr)
    {
        XcOperatorObj minusOp = new XcOperatorObj(XcOperatorEnum.MINUS, new XcExprObj[2]);
        minusOp.addChild(expr);
        minusOp.addChild(_one);

        return minusOp;
    }

    private static XcCompoundValueObj _createSubArrayDimensionType(XcXmpSubArrayRefObj subArrayDim)
    {
        XcCompoundValueObj obj = new XcCompoundValueObj();
        obj.addChild(subArrayDim.getLowerBound());
        if(subArrayDim.getUpperBound() == null) {
            obj.addChild(_createMinusOne(subArrayDim.getArraySize()));
        } else {
            obj.addChild(subArrayDim.getUpperBound());
        }
        obj.addChild(subArrayDim.getStep());
        obj.addChild(subArrayDim.getArraySize());
        obj.addChild(_createSizeOf(subArrayDim.getUnitType()));

        return obj;
    }

    /**
     * Sets and Gets temporary variable for XcXmpCoArrayRefObj
     * which will be translated to __xmp_coarray_get function call.
     */
    private static void _getCoArrayTmpVar(XcNode node, XcDeclsObj decls)
    {
        if(node == null)
            return;

        boolean isAssignExpr = false;

        if((node instanceof XcExprObj) == false)
            return;
        else
            _getCoArrayTmpVarFromExpr((XcExprObj)node, decls);

        if(node instanceof XcOperatorObj)
            isAssignExpr = ((XcOperatorObj)node).isAssignExpr();

        if(node instanceof XcXmpCoArrayAssignObj)
            isAssignExpr = true;

        XcNode[] nodes = node.getChild();
        if(nodes == null)
            return;

        for(XcNode child : nodes) {
            if(isAssignExpr == true) {
                isAssignExpr = false;
                continue;
            }

            _getCoArrayTmpVar(child, decls);
        }
    }

    /**
     * Gets and Sets temp variables to XcXmpCoArrayRefObjs.
     * 
     * @param expr
     * @param decls
     */
    private static void _getCoArrayTmpVarFromExpr(XcExprObj expr, XcDeclsObj decls)
    {
        if(expr instanceof XcXmpCoArrayRefObj) {
            XcXmpCoArrayRefObj coArrayRef = (XcXmpCoArrayRefObj)expr;

            XcType elementType = coArrayRef.getElementType();

            XcIdent ident = new XcIdent(_createTmpValueName());
            ident.setType(elementType);

            coArrayRef.setTempVar(ident);

            XcDeclObj decl = new XcDeclObj(ident);
            decls.addChild(decl);
        }
    }
}
