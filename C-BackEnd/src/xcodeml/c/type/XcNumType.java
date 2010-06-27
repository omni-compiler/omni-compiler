/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * type of number
 */
public abstract class XcNumType extends XcBaseType
{
    private int _bitField;
    
    public XcNumType(XcBaseTypeEnum basicTypeEnum, String typeId)
    {
        super(basicTypeEnum, typeId);
    }
    
    public int getBitField()
    {
        return _bitField;
    }
    
    public void setBitField(int bitField)
    {
        _bitField = bitField;
    }
}
