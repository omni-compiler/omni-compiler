/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.binding.IXbConstant;
import xcodeml.binding.IXbStringContent;


/**
 * Concrete classes implement this interface are follows:
 * XbcIntConstant, XbcFloatConstant, XbcLonglongConstant
 * XbcStringConstant, XbcMoeConstant
 */
public interface IXbcConstant extends IXbConstant, IXbcTypedExpr
{
}

