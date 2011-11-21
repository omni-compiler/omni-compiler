/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import org.w3c.dom.Node;

import xcodeml.f.util.XmfNodeVisitorMap;
import xcodeml.util.XmDomUtil;

import static xcodeml.f.util.XmfNodeVisitorMap.Pair;

/**
 * XcodeML/F validation utility, for DOM node.
 */
public class XfRuntimeDomValidator {
    private String errorDescription;

    private XmfNodeVisitorMap<NodeValidator> visitorMap;

    @SuppressWarnings("unchecked")
	public XfRuntimeDomValidator() {
        visitorMap = new XmfNodeVisitorMap<NodeValidator>(pairs);
    }

    public final boolean validateAttr(Node node) {
        NodeValidator visitor = visitorMap.getVisitor(node.getNodeName());
        if (visitor == null) {
            return true;
        } else {
            return visitor.validateAttr(node);
        }
    }

    /**
     * Get error description;
     */
    public String getErrDesc()
    {
        return errorDescription;
    }

    private abstract class NodeValidator {
        public abstract boolean validateAttr(Node n);
    }

    // FbasicType
    class FbasicTypeValidator extends NodeValidator {
        /**
         * Validate attributes of 'FbasicType'
         *
         * @param obj obj is validated aboute attributes.
         *
         * @return false if there are insufficient attributes.
         *
         */
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "type"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "type",
                                             n.getNodeName());
                return false;
            }

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "ref"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "ref",
                                             n.getNodeName());

                return false;
            }

            return true;
        }
    }

    // FfunctionType
    class FfunctionTypeValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "type"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "type",
                                             n.getNodeName());
                return false;
            }

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "return_type"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "return_type",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FstructType
    class FstructTypeValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "type"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "type",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FmoduleDefinition
    class FmoduleDefinitionValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FblockDataDefinition
    class FblockDataDefinitionValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FuseDecl
    class FuseDeclValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FuseOnlyDecl
    class FuseOnlyDeclValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // statementLabel
    class StatementLabelValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "label_name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "label_name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FprintStatement
    class FprintStatementValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "format"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "format",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FformatDecl
    class FformatDeclValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "format"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "format",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // FmemberRef
    class FmemberRefValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "member"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "member",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // id
    class IdValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "sclass"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "sclass",
                                             n.getNodeName());
                return false;
            }

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "type"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "type",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // name
    class NameValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "type"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "type",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // rename
    class RenameValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "use_name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "use_name",
                                             n.getNodeName());
                return false;
            }

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "local_name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "local_name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // renamable
    class RenamableValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "use_name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "use_name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
    }

    // namedValue
    class NamedValueValidator extends NodeValidator {
        @Override public boolean validateAttr(Node n) {
            if (n == null)
                throw new IllegalArgumentException();

            if (XfUtil.isNullOrEmpty(XmDomUtil.getAttr(n, "name"))) {
                errorDescription =
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "name",
                                             n.getNodeName());
                return false;
            }

            return true;
        }
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
    @SuppressWarnings("unchecked")
    private Pair[] pairs = {
        new Pair("FbasicType", new FbasicTypeValidator()),
        new Pair("FfunctionType", new FfunctionTypeValidator()),
        new Pair("FstructType", new FstructTypeValidator()),
        new Pair("FmoduleDefinition", new FmoduleDefinitionValidator()),
        new Pair("FuseDecl", new FuseDeclValidator()),
        new Pair("FuseOnlyDecl", new FuseOnlyDeclValidator()),
        new Pair("statementLabel", new StatementLabelValidator()),
        new Pair("FprintStatement", new FprintStatementValidator()),
        new Pair("FformatDecl", new FformatDeclValidator()),
        new Pair("FmemberRef", new FmemberRefValidator()),
        // These methods are not used at XfRuntimeValidator.
        //new Pair("FblockDataDefinition", new FblockDataDefinitionValidator()),
        //new Pair("id", new IdValidator()),
        //new Pair("name", new NameValidator()),
        //new Pair("rename", new RenameValidator()),
        //new Pair("renamable", new RenamableValidator()),
        //new Pair("namedValue", new NamedValueValidator()),
    };
}
