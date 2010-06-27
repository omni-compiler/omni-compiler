/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.binding;

import xcodeml.f.binding.gen.XbfNamedValueList;

public interface IXbfIOStatement
{
    public XbfNamedValueList getNamedValueList();
    
    public void setNamedValueList(XbfNamedValueList namedValueList);
}
