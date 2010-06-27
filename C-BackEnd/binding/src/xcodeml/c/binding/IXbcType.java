/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.c.binding.gen.XbcGccAttributes;

/**
 * Concreate classes implement the interface are follows : 
 * XbcBasicType, XbcPointerType, XbcFunctionType
 * XbcArrayType, XbcStructType, XbcUnionType, XbcEnumType
 */
public interface IXbcType
{
    public String getType();
    
    public void setType(String s);
    
    /**
     * Gets a content of is_volatile attribute.
     *
     * @return a content of is_volatile attribute.
     */
    public String getIsVolatile();

    public void setIsVolatile(String enabled);
    
    /**
     * Gets a content of is_const attribute.
     *
     * @return a content of is_const attribute.
     */
    public String getIsConst();

    public void setIsConst(String enabled);
    
    /**
     * Gets a content of is_restrict attribute.
     *
     * @return a content of is_restrict attribute.
     */
    public String getIsRestrict();

    public void setIsRestrict(String enabled);

    /**
     * Gets the XbcGccAttributes property <b>gccAttributes</b>.
     *
     * @return XbcGccAttributes
     */
    public XbcGccAttributes getGccAttributes();

    public void setGccAttributes(XbcGccAttributes attrs);
}
