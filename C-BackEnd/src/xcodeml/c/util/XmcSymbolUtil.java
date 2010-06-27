/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.util;

import xcodeml.XmObj;
import xcodeml.c.binding.gen.IXbcSymbolsChoice;
import xcodeml.c.binding.gen.XbcCompoundStatement;
import xcodeml.c.binding.gen.XbcFunctionDefinition;
import xcodeml.c.binding.gen.XbcId;
import xcodeml.c.binding.gen.XbcXcodeProgram;

/**
 * Utilities for XcodeML object symbol.
 */
public class XmcSymbolUtil
{

    /**
     * Looks up identifier symbol.
     */
    public static XbcId lookupSymbol(XmObj obj, String name) {
        return doLookup(obj, name, null);
    }

    /**
     * Looks up tagname symbol.
     *
     * @param xobj object binding to XcodeML element include name.
     * @param name represents symbol.
     * @return an identifier of the name.
     */
    public static XbcId lookupTagname(XmObj obj, String name) {
        return doLookup(obj, name, XbcId.SCLASS_TAGNAME);
    }
    
    /**
     * Looks up label symbol.
     *
     * @param xobj object binding to XcodeML element include name.
     * @param name represents symbol.
     * @return an identifier of the name.
     */
    public static XbcId lookupLabel(XmObj obj, String name) {
        return doLookup(obj, name, XbcId.SCLASS_LABEL);
    }
    
    private static XbcId doLookup(XmObj obj, String name, String sclass)
    {
        if(obj == null) {
            return null;
        }

        XbcId result = null;
        if(obj instanceof XbcCompoundStatement) {
            XbcCompoundStatement stmt = (XbcCompoundStatement)obj;
            if(stmt.getSymbols() != null)
                result = doLookup(stmt.getSymbols().getContent(), name, sclass);
        } else if(obj instanceof XbcFunctionDefinition) {
            XbcFunctionDefinition def = (XbcFunctionDefinition)obj;
            if(def.getSymbols() != null)
                result = doLookup(def.getSymbols().getContent(), name, sclass);
        } else if(obj instanceof XbcXcodeProgram) {
            XbcXcodeProgram prog = (XbcXcodeProgram)obj;
            if(prog.getGlobalSymbols() != null)
                result = doLookup(prog.getGlobalSymbols().getId(), name, sclass);
        }

        if(result != null) {
            return result;
        } else {
            return lookupSymbol((XmObj)obj.getParent(), name);
        }
    }

    private static XbcId doLookup(IXbcSymbolsChoice[] objs, String name, String sclass)
    {
        if(name == null) {
            throw new NullPointerException("name is null");
        }
        
        for(IXbcSymbolsChoice obj : objs) {
            if(obj instanceof XbcId) {
                XbcId id = (XbcId)obj;
                String idname = id.getName().getContent();

                if(name.equals(idname)) {
                    if (sclass == null || sclass.equals(id.getSclass())) {
                        return id;
                    }
                }
            }
        }
        return null;
    }

}
