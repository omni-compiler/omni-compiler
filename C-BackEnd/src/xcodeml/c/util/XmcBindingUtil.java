/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.util;

import xcodeml.XmException;
import xcodeml.c.binding.gen.XbcLonglongConstant;
import xcodeml.c.decompile.XcConstObj;
import xcodeml.c.type.XcBaseTypeEnum;
import xcodeml.util.XmBindingException;
import xcodeml.util.XmStringUtil;

public class XmcBindingUtil
{
    public static XcConstObj.LongLongConst createLongLongConst(XbcLonglongConstant visitable)
    {
        String text = XmStringUtil.trim(visitable.getContent());
        if(text == null)
            throw new XmBindingException(visitable, "invalid constant value");
        String[] values = text.split(" ");
        if(values.length != 2)
            throw new XmBindingException(visitable, "invalid constant value");

        XcBaseTypeEnum btEnum;
        String typeId = XmStringUtil.trim(visitable.getType());
        if(typeId == null)
            btEnum = XcBaseTypeEnum.LONGLONG;
        else
            btEnum = XcBaseTypeEnum.getByXcode(typeId);

        if(btEnum == null)
            throw new XmBindingException(visitable, "invalid type '" + typeId + "' as long long constant");
        
        XcConstObj.LongLongConst obj;

        try {
            obj = new XcConstObj.LongLongConst(values[0], values[1], btEnum);
        } catch(XmException e) {
            throw new XmBindingException(visitable, e.getMessage());
        }
        
        return obj;
    }

	public static XcConstObj.LongLongConst createLongLongConst(String valuesText,
															   String typeId) {
		String[] values = XmStringUtil.trim(valuesText).split(" ");
		if (values.length != 2)
			return null;

		XcBaseTypeEnum btEnum;
		if (typeId == null)
			btEnum = XcBaseTypeEnum.LONGLONG;
		else
			btEnum = XcBaseTypeEnum.getByXcode(typeId);

		if (btEnum == null)
			return null;

		XcConstObj.LongLongConst obj;

		try {
			obj = new XcConstObj.LongLongConst(values[0], values[1], btEnum);
		} catch(XmException e) {
			return null;
		}
		return obj;
	}
}
