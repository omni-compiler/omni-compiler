/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

public enum XcOperatorEnum
{
    /**
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
     *  (sizeof, alignof operators)
     *      sizeOfExpr, addrOfExpr, gccAlignOfExpr
     *  (label operator)
     *      labelAddr
     */

    UNARY_MINUS     (XcOperatorTypeEnum.UNARY_PRE, "-"),
    BIT_NOT         (XcOperatorTypeEnum.UNARY_PRE, "~"),
    LOG_NOT         (XcOperatorTypeEnum.UNARY_PRE, "!"),
    PRE_INCR        (XcOperatorTypeEnum.UNARY_PRE, "++"),
    PRE_DECR        (XcOperatorTypeEnum.UNARY_PRE, "--"),
    POST_INCR       (XcOperatorTypeEnum.UNARY_POST, "++"),
    POST_DECR       (XcOperatorTypeEnum.UNARY_POST, "--"),
    ASSIGN          (XcOperatorTypeEnum.ASSIGN, "="),
    ASSIGN_PLUS     (XcOperatorTypeEnum.ASSIGN, "+="),
    ASSIGN_MINUS    (XcOperatorTypeEnum.ASSIGN, "-="),
    ASSIGN_MUL      (XcOperatorTypeEnum.ASSIGN, "*="),
    ASSIGN_DIV      (XcOperatorTypeEnum.ASSIGN, "/="),
    ASSIGN_MOD      (XcOperatorTypeEnum.ASSIGN, "%="),
    ASSIGN_LSHIFT   (XcOperatorTypeEnum.ASSIGN, "<<="),
    ASSIGN_RSHIFT   (XcOperatorTypeEnum.ASSIGN, ">>="),
    ASSIGN_BIT_AND  (XcOperatorTypeEnum.ASSIGN, "&="),
    ASSIGN_BIT_OR   (XcOperatorTypeEnum.ASSIGN, "|="),
    ASSIGN_BIT_XOR  (XcOperatorTypeEnum.ASSIGN, "^="),
    PLUS            (XcOperatorTypeEnum.BINARY, "+"),
    MINUS           (XcOperatorTypeEnum.BINARY, "-"),
    MUL             (XcOperatorTypeEnum.BINARY, "*"),
    DIV             (XcOperatorTypeEnum.BINARY, "/"),
    MOD             (XcOperatorTypeEnum.BINARY, "%"),
    LSHIFT          (XcOperatorTypeEnum.BINARY, "<<"),
    RSHIFT          (XcOperatorTypeEnum.BINARY, ">>"),
    BIT_AND         (XcOperatorTypeEnum.BINARY, "&"),
    BIT_OR          (XcOperatorTypeEnum.BINARY, "|"),
    BIT_XOR         (XcOperatorTypeEnum.BINARY, "^"),
    LOG_EQ          (XcOperatorTypeEnum.LOG, "=="),
    LOG_NEQ         (XcOperatorTypeEnum.LOG, "!="),
    LOG_GE          (XcOperatorTypeEnum.LOG, ">="),
    LOG_GT          (XcOperatorTypeEnum.LOG, ">"),
    LOG_LE          (XcOperatorTypeEnum.LOG, "<="),
    LOG_LT          (XcOperatorTypeEnum.LOG, "<"),
    LOG_AND         (XcOperatorTypeEnum.LOG, "&&"),
    LOG_OR          (XcOperatorTypeEnum.LOG, "||"),
    COND            (XcOperatorTypeEnum.COND, "?"),
    COMMA           (XcOperatorTypeEnum.COMMA, ","),
    SIZEOF          (XcOperatorTypeEnum.SIZEOF, "sizeof"),
    ADDROF          (XcOperatorTypeEnum.SIZEOF, "&"),
    ALIGNOF         (XcOperatorTypeEnum.SIZEOF, "__alignof__"),
    LABELADDR       (XcOperatorTypeEnum.LABELADDR, "&&"),
    ;
    
    private XcOperatorTypeEnum _opeType;
    
    private String _code;
    
    private XcOperatorEnum(XcOperatorTypeEnum opeType, String code)
    {
        _opeType = opeType;
        _code = code;
    }
    
    public final XcOperatorTypeEnum getOperatorType()
    {
        return _opeType;
    }
    
    public final String getCode()
    {
        return _code;
    }
}
