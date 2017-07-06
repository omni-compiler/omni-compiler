package xcodeml.c.util;

import xcodeml.util.XmException;
import xcodeml.c.decompile.XcConstObj;
import xcodeml.c.type.XcBaseTypeEnum;
import xcodeml.util.XmStringUtil;

public class XmcBindingUtil
{
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
