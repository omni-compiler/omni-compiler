/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.binding;

import xcodeml.f.binding.gen.XbfNamedValueList;
import xcodeml.f.binding.gen.XbfValueList;

public interface IXbfRWStatement
{
    public XbfNamedValueList getNamedValueList();
    
    public void setNamedValueList(XbfNamedValueList namedValueList);

    public XbfValueList getValueList();
    
    public void setValueList(XbfValueList valueList);
}
