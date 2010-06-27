/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.c.binding.gen.XbcArraySize;

/**
 * Concrete classes implement this interface are follows :
 * XbcArrayType, XbcCoArrayType
 */
public interface IXbcArrayType
{
    /**
     * Gets array_size attribute.
     *
     * @return contents of array_size attribute.
     */
    public String getArraySize1();
    public void setArraySize1(String size1);

    /**
     * Gets arraySize tag element.
     *
     * @return arraySize tag object.
     */
    public XbcArraySize getArraySize2();
    public void setArraySize2(XbcArraySize size2);

    /**
     * Gets element_type attribute.
     *
     * @return content of element_type attribute.
     */
    public String getElementType();
    public void setElementType(String elemType);
}
