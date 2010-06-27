/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import xcodeml.binding.IRNode;
import xcodeml.f.binding.gen.*;

/**
 * XcodeML/F validation utility.
 */
public class XfRuntimeValidator
{
    private String errorDescription;

    /**
     * Get error description;
     */
    public String getErrDesc()
    {
        return errorDescription;
    }

    /**
     * Validate attributes of 'FbasicType'
     *
     * @param obj obj is validated aboute attributes.
     * 
     * @return false if there are insufficient attributes.
     *
     */
    public final boolean validAttr(XbfFbasicType obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getType())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "type",
                                   XfUtil.getElementName(obj));
            return false;
        }

        if (XfUtil.isNullOrEmpty(obj.getRef())) {
            errorDescription =
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "ref",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFfunctionType obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getType())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "type",
                                   XfUtil.getElementName(obj));
            return false;
        }

        if (XfUtil.isNullOrEmpty(obj.getReturnType())) {
            errorDescription =
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "return_type",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFstructType obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getType())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "type",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFmoduleDefinition obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFblockDataDefinition obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFuseDecl obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFuseOnlyDecl obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfStatementLabel obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getLabelName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "label_name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFprintStatement obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getFormat())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "format",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFformatDecl obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getFormat())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "format",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfFmemberRef obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getMember())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "member",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfId obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getSclass())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "sclass",
                                   XfUtil.getElementName(obj));
            return false;
        }

        if (XfUtil.isNullOrEmpty(obj.getType())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "type",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfName obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getType())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "type",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfRename obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getUseName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "use_name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        if (XfUtil.isNullOrEmpty(obj.getLocalName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "local_name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfRenamable obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getUseName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "use_name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }

    public final boolean validAttr(XbfNamedValue obj)
    {
        if (obj == null)
            throw new IllegalArgumentException();

        if (XfUtil.isNullOrEmpty(obj.getName())) {
            errorDescription = 
                XfUtil.formatError(obj,
                                   XfError.XCODEML_NEED_ATTR,
                                   "name",
                                   XfUtil.getElementName(obj));
            return false;
        }

        return true;
    }


    /* NOTE: 
     * Required attributes are follows.
     *
     * FbasicType.type
     * FbasicType.ref
     * FfunctionType.type
     * FfunctionType.return_type
     * FstructType.type
     * FmoduleDefinition.name
     * FuseDecl.name
     * FuseOnlyDecl.name
     * StatementLabel.label_name
     * FPrintStatement.format
     * FformatDecl.format
     * FmemberRef.member
     * id.sclass
     * id.type
     * name.type
     * rename.use_name
     * rename.local_name
     * renamable.use_name
     * renamable.local_name
     * namedValue.name
     * (anyStatement).lineno
     * (anyStatement).file
    */
    public final boolean validAttr(IRNode node)
    {
        if(node instanceof XbfFbasicType)
            return validAttr((XbfFbasicType)node);
        if(node instanceof XbfFfunctionType)
            return validAttr((XbfFfunctionType)node);
        if(node instanceof XbfFstructType)
            return validAttr((XbfFstructType)node);
        if(node instanceof XbfFmoduleDefinition)
            return validAttr((XbfFmoduleDefinition)node);
        if(node instanceof XbfFuseDecl)
            return validAttr((XbfFuseDecl)node);
        if(node instanceof XbfFuseOnlyDecl)
            return validAttr((XbfFuseOnlyDecl)node);
        if(node instanceof XbfStatementLabel)
            return validAttr((XbfStatementLabel)node);
        if(node instanceof XbfFprintStatement)
            return validAttr((XbfFprintStatement)node);
        if(node instanceof XbfFformatDecl)
            return validAttr((XbfFformatDecl)node);
        if(node instanceof XbfFmemberRef)
            return validAttr((XbfFmemberRef)node);

        return true;
    }
}