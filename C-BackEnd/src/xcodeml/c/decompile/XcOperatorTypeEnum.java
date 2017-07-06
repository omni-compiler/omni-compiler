package xcodeml.c.decompile;

public enum XcOperatorTypeEnum
{
    UNARY_POST  (1),
    UNARY_PRE   (1),
    BINARY      (2),
    ASSIGN      (2),
    LOG         (2),
    COND        (3),
    SIZEOF      (1),
    LABELADDR   (1),
    COMMA       (-1),
    ;

    /* number of XmExprObj as arguments. -1 means any. */
    private int _numOfExprs;
    
    private XcOperatorTypeEnum(int numOfExprs)
    {
        _numOfExprs = numOfExprs;
    }
    
    public final int getNumOfExprs()
    {
        return _numOfExprs;
    }
}
