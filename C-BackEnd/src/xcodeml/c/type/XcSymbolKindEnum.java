/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;


public enum XcSymbolKindEnum
{
    VAR         (XcIdentTableEnum.MAIN),
    FUNC        (XcIdentTableEnum.MAIN),
    MOE         (XcIdentTableEnum.MAIN),
    TYPE        (XcIdentTableEnum.MAIN),
    TAGNAME     (XcIdentTableEnum.TAGNAME),
    LABEL       (XcIdentTableEnum.LABEL),
    ANONYM      (XcIdentTableEnum.ANONYM),
    ;
    
    private XcIdentTableEnum _iteEnum;
    
    private XcSymbolKindEnum(XcIdentTableEnum iteEnum)
    {
        _iteEnum = iteEnum;
    }
    
    public XcIdentTableEnum getIdentTableEnum()
    {
        return _iteEnum;
    }
}
