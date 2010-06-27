/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import xcodeml.XmObj;
import xcodeml.c.binding.gen.XbcId;
import xcodeml.c.util.XmcSymbolUtil;
import xcodeml.f.binding.gen.XbfId;
import xcodeml.f.util.XmfSymbolUtil;
import xcodeml.util.XmOption;

public class XmSymbolUtil
{
    /**
     * find symbol in specified context.
     */
    public static XmSymbol lookupSymbol(XmObj context, String name)
    {
        if(name == null) {
            throw new NullPointerException("name is null");
        }
        
        XmSymbol symbol = null;
        XbcId xcid = null;
        XbfId xfid = null;
        
        switch(XmOption.getLanguage()) {
        case C:
            xcid = XmcSymbolUtil.lookupSymbol(context, name);
            if(xcid != null) {
                symbol = new XmSymbol(xcid);
            }
            break;
        case F:
            xfid = XmfSymbolUtil.lookupSymbol(context, name);
            if(xfid != null) {
                symbol = new XmSymbol(xfid);
            }
            break;
        }
        
        return symbol;
    }
}
