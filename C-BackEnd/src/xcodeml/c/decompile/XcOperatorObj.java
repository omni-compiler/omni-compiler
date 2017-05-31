/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *  (prefix unary operators)
 *      unaryMinusExpr, bitNotExpr, logNotExpr
 *      [++, --]
 *  (post unary operators)
 *      postIncrExpr, postDecrExpr
 *  (assign operators)
 *      assignExpr,
 *      asgPlusExpr, asgMinusExpr, asgMulExpr, asgDivExpr, asgModExpr,
 *      asgLshiftExpr, asgRshiftExpr, asgBitAndExpr, asgBitOrExpr, asgBitXorExpr,
 *  (calculation operators)
 *      plusExpr, minusExpr, mulExpr, divExpr, modExpr,
 *      LshiftExpr, RshiftExpr, bitAndExpr, bitOrExpr, bitXorExpr
 *  (comparison operators)
 *      logEQExpr, logNEQExpr, logGEExpr, logGTExpr, logLEExpr, logLTExpr,
 *      logAndExpr, logOrExpr
 *  (ternary operators)
 *      condExpr
 *  (comma operator)
 *      commaExpr
 *  (label operator)
 *      labelAddr
 */
public class XcOperatorObj extends XcObj implements XcExprObj, XcXmpCoArrayParent
{
    private XcOperatorEnum _opeEnum;

    private XcExprObj[] _exprs;

    /**
     * Creates XcOperatorObj
     */
    public XcOperatorObj()
    {
    }

    /**
     * Creates XcOperatorObj
     * 
     * @param opeEnum indicates which operator is the object.
     */
    public XcOperatorObj(XcOperatorEnum opeEnum)
    {
        _opeEnum = opeEnum;
    }

    /**
     * Creates XcOperatorObj.
     * 
     * @param opeEnum indicates which operator is the object.
     * @param exprs terms of the operator.
     */
    public XcOperatorObj(XcOperatorEnum opeEnum, XcExprObj[] exprs)
    {
        int numOfExprs = opeEnum.getOperatorType().getNumOfExprs();
        if(exprs == null || numOfExprs > 0 && exprs.length != numOfExprs)
            throw new IllegalArgumentException(
                "number of expression for the operator is invalid : " + (exprs == null ? "0" : exprs.length));

        _exprs = exprs;
        _opeEnum = opeEnum;
    }

    /**
     * Tests if is the operator an assignment operator.
     * 
     * @return true if the operator is an assignment operator.
     */
    public boolean isAssignExpr()
    {
        if(XcOperatorTypeEnum.ASSIGN.equals(_opeEnum.getOperatorType())) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Creates XcOperatorObj.
     * 
     * @param opeEnum indicates which operator is the object.
     * @param expr a term of the operator.
     */
    public XcOperatorObj(XcOperatorEnum opeEnum, XcExprObj expr) throws XmException
    {
        this(opeEnum, new XcExprObj[] { expr } );
    }

    /**
     * Creates XcOperatorObj.
     * 
     * @param opeEnum indicates which operator is the object.
     * @param expr1 the first term of the operator.
     * @param expr2 the second term of the operator.
     */
    public XcOperatorObj(XcOperatorEnum opeEnum, XcExprObj expr1, XcExprObj expr2) throws XmException
    {
        this(opeEnum, new XcExprObj[] { expr1, expr2} );
    }

    /**
     * Creates XcOperatorObj.
     * 
     * @param opeEnum indicates which operator is the object.
     * @param expr1 the first term of the operator.
     * @param expr2 the second term of the operator.
     * @param expr2 the third term of the operator.
     */
    public XcOperatorObj(XcOperatorEnum opeEnum, XcExprObj expr1, XcExprObj expr2, XcExprObj expr3) throws XmException
    {
        this(opeEnum, new XcExprObj[] { expr1, expr2, expr3 } );
    }

    /**
     * Gets a enumerator of the operator. 
     * 
     * @return a enumerator of the operator. 
     */
    public final XcOperatorEnum getOperatorEnum()
    {
        return _opeEnum;
    }

    /**
     * Gets the terms of the operator.
     * 
     * @return the terms of the operator.
     */
    public final XcExprObj[] getExprObjs()
    {
        return _exprs;
    }

    /**
     * Interal object represents labelAddr.
     */
    public class LabelAddrExpr extends XcOperatorObj
    {
        private String _label;

        /**
         * Gets a label string.
         * 
         * @return a label string.
         */
        public String getLabel()
        {
            return _label;
        }

        /**
         * Creates LabelAddrExpr
         * 
         * @param label a label string.
         */
        public LabelAddrExpr(String label)
        {
            super(XcOperatorEnum.LABELADDR);
            _label = label;
        }

        @Override
        public void checkChild()
        {
            if(_label == null)
                throw new IllegalArgumentException(
                "number of expression for the operator is invalid : 0");
        }
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj) {

            if(_exprs == null) {
                _exprs = new XcExprObj[_opeEnum.getOperatorType().getNumOfExprs()];
            }

            for(int i = 0; i < _exprs.length; ++i) {
                if(_exprs[i] == null) {
                    _exprs[i] = (XcExprObj)child;
                    return;
                }
            }

            throw new IllegalArgumentException("too many terms : " + child.getClass().getName());
        } else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
        if (this instanceof LabelAddrExpr) {
            ((LabelAddrExpr)this).checkChild();
        } else {
            int numOfExprs = _opeEnum.getOperatorType().getNumOfExprs(); 
            if(_exprs == null || numOfExprs > 0 && _exprs.length != numOfExprs)
                throw new IllegalArgumentException(
                "number of expression for the operator is invalid : " + (_exprs == null ? "0" : _exprs.length));
        }
    }

    @Override
    public XcNode[] getChild()
    {
        if(_exprs == null)
            return null;
        switch(_exprs.length)
        {
        case 1:
            return new XcNode[] { _exprs[0] };
        case 2:
            return new XcNode[] { _exprs[0], _exprs[1] };
        default:
            return new XcNode[] { _exprs[0], _exprs[1], _exprs[2] };
        }
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
        case 1:
        case 2:
            _exprs[index] = (XcExprObj)child;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        String opeCode = _opeEnum.getCode();

        switch(_opeEnum.getOperatorType()) {
        case UNARY_PRE:
            w.add(opeCode).addBraceIfNeeded(_exprs[0]);
            break;
        case UNARY_POST:
            w.spc().addBraceIfNeeded(_exprs[0]).add(opeCode);
            break;
        case ASSIGN:
        case BINARY:
        case LOG:
            w.spc().addBraceIfNeeded(_exprs[0]).addSpc(opeCode).spc().addBraceIfNeeded(_exprs[1]);
            break;
        case COND:
            if (_exprs[2] == null) {
                w.spc().addBraceIfNeeded(_exprs[0]).addSpc(opeCode)
                    .addSpc(":").spc().addBraceIfNeeded(_exprs[1]);
            } else {
                w.spc().addBraceIfNeeded(_exprs[0]).addSpc(opeCode)
                    .addBraceIfNeeded(_exprs[1]).addSpc(":").spc().addBraceIfNeeded(_exprs[2]);
            }
            break;
        case COMMA:
            w.add("(");
            for(int i = 0; i < _exprs.length; ++i) {
                if(i > 0)
                    w.add(opeCode).spc();
                w.add(_exprs[i]);
            }
            w.add(")");
            break;
        case SIZEOF:
            if (this instanceof XcSizeOfExprObj) {
                w.add(opeCode).addBrace(((XcSizeOfExprObj)this).getTypeName());
            } else {
                w.spc().add(opeCode).addBrace(_exprs[0]);
            }
            break;
        case  LABELADDR:
            w.add(opeCode).add(((LabelAddrExpr)this).getLabel());
            break;
        }
    }

    @Override
    public void setCoArrayContent(XcExprObj expr)
    {
        _exprs[0] = expr;
    }
}
