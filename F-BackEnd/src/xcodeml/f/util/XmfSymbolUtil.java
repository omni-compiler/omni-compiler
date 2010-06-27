/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.util;

import xcodeml.XmObj;
import xcodeml.f.binding.gen.XbfFblockDataDefinition;
import xcodeml.f.binding.gen.XbfFfunctionDefinition;
import xcodeml.f.binding.gen.XbfFmoduleDefinition;
import xcodeml.f.binding.gen.XbfId;
import xcodeml.f.binding.gen.XbfXcodeProgram;

/**
 * Utilities for XcodeML object symbol.
 */
public class XmfSymbolUtil
{
    /**
     * Looks up identifier symbol.
     */
    public static XbfId lookupSymbol(XmObj obj, String name) {
        return doLookup(obj, name, null);
    }

    /**
     * Looks up type name symbol.
     *
     * @param xobj object binding to XcodeML element include name.
     * @param name represents symbol.
     * @return an identifier of the name.
     */
    public static XbfId lookupTypename(XmObj obj, String name) {
        return doLookup(obj, name, XbfId.SCLASS_FTYPE_NAME);
    }
    
    /**
     * Looks up common name symbol.
     *
     * @param xobj object binding to XcodeML element include name.
     * @param name represents symbol.
     * @return an identifier of the name.
     */
    public static XbfId lookupLabel(XmObj obj, String name) {
        return doLookup(obj, name, XbfId.SCLASS_FCOMMON_NAME);
    }
    
    private static XbfId doLookup(XmObj obj, String name, String sclass)
    {
        if(obj == null) {
            return null;
        }

        XbfId result = null;
        if(obj instanceof XbfFfunctionDefinition) {
            XbfFfunctionDefinition def = (XbfFfunctionDefinition)obj;
            if(def.getSymbols() != null)
                result = doLookup(def.getSymbols().getId(), name, sclass);
        } else if(obj instanceof XbfFmoduleDefinition) {
            XbfFmoduleDefinition def = (XbfFmoduleDefinition)obj;
            if(def.getSymbols() != null)
                result = doLookup(def.getSymbols().getId(), name, sclass);
        } else if(obj instanceof XbfFblockDataDefinition) {
            XbfFblockDataDefinition def = (XbfFblockDataDefinition)obj;
            if(def.getSymbols() != null)
                result = doLookup(def.getSymbols().getId(), name, sclass);
        } else if(obj instanceof XbfXcodeProgram) {
            XbfXcodeProgram prog = (XbfXcodeProgram)obj;
            if(prog.getGlobalSymbols() != null)
                result = doLookup(prog.getGlobalSymbols().getId(), name, sclass);
        }

        if(result != null) {
            return result;
        } else {
            return lookupSymbol((XmObj)obj.getParent(), name);
        }
    }

    private static XbfId doLookup(XbfId[] ids, String name, String sclass)
    {
        if(name == null) {
            throw new NullPointerException("name is null");
        }
        
        for(XbfId id : ids) {
            String idname = id.getName().getContent();

            if(name.equals(idname)) {
                if (sclass == null || sclass.equals(id.getSclass())) {
                    return id;
                }
            }
        }
        return null;
    }
}
