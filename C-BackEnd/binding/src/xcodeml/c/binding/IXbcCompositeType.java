/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.c.binding.gen.XbcSymbols;

/**
 * Concrete classes implement this interface are follows:
 * XbcStructType, XbcUnionType.
 */
public interface IXbcCompositeType extends IXbcType
{
    /**
     * Gets symbols tag object.
     *
     * @return symbols tag object.
     */
    public XbcSymbols getSymbols();
}
