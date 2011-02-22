/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.util.Vector;

public class XMPtranslateGlobalPragma {
  private XMPglobalDecl		_globalDecl;

  public XMPtranslateGlobalPragma(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }

  public void translate(Xobject x) {
    try {
      translatePragma(x);
    } catch (XMPexception e) {
      XMP.error(x.getLineNo(), e.getMessage());
    }
  }

  public void translatePragma(Xobject x) throws XMPexception {
    String pragmaName = x.getArg(0).getString();

    switch (XMPpragma.valueOf(pragmaName)) {
      case NODES:
        translateNodes(x);
        break;
      case TEMPLATE:
        translateTemplate(x);
        break;
      case DISTRIBUTE:
        translateDistribute(x);
        break;
      case ALIGN:
        translateAlign(x);
        break;
      case SHADOW:
        translateShadow(x);
        break;
      case COARRAY:
        translateCoarray(x);
        break;
      default:
        throw new XMPexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }

  private void translateNodes(Xobject nodesPragma) throws XMPexception {
    XMPnodes.translateNodes((XobjList)nodesPragma.getArg(1), _globalDecl, false, null);
  }

  private void translateTemplate(Xobject templatePragma) throws XMPexception {
    XMPtemplate.translateTemplate((XobjList)templatePragma.getArg(1), _globalDecl, false, null);
  }

  private void translateDistribute(Xobject distributePragma) throws XMPexception {
    XMPtemplate.translateDistribute((XobjList)distributePragma.getArg(1), _globalDecl, false, null);
  }

  private void translateAlign(Xobject alignPragma) throws XMPexception {
    XMPalignedArray.translateAlign((XobjList)alignPragma.getArg(1), _globalDecl, false, null);
  }

  private void translateShadow(Xobject shadowPragma) throws XMPexception {
    XMPshadow.translateShadow((XobjList)shadowPragma.getArg(1), _globalDecl, false, null);
  }

  private void translateCoarray(Xobject coarrayPragma) throws XMPexception {
    XobjList coarrayDecl = (XobjList)coarrayPragma.getArg(1);

    String coarrayName = coarrayDecl.getArg(0).getString();
    if(_globalDecl.getXMPcoarray(coarrayName) != null) {
      throw new XMPexception("coarray " + coarrayName + " is already declared");
    }

    // FIXME allow an aligned array to be a coarray? check the specifications
    if (_globalDecl.getXMPalignedArray(coarrayName) != null) {
      throw new XMPexception("an aligned array cannot be declared as a coarray");
    }

    Ident varId = _globalDecl.findVarIdent(coarrayName);
    if (varId == null) {
      throw new XMPexception("coarray '" + coarrayName + "' is not declared");
    }

    Xtype elmtType = null;
    Xtype varType = varId.Type();
    if (varType.getKind() == Xtype.ARRAY) {
      elmtType = varType.getArrayElementType();
    }
    else {
      elmtType = varType;
    }

    // decl descriptors
    Ident XMPdescId = _globalDecl.declStaticIdent(XMP.CAF_DESC_PREFIX_ + coarrayName, Xtype.voidPtrType);
    Ident commDescId = null;
    if (varId.getStorageClass() == StorageClass.EXTERN) {
      commDescId = _globalDecl.declExternIdent(XMP.CAF_COMM_PREFIX_ + coarrayName, Xtype.voidPtrType);
    }
    else if (varId.getStorageClass() == StorageClass.STATIC) {
      commDescId = _globalDecl.declStaticIdent(XMP.CAF_COMM_PREFIX_ + coarrayName, Xtype.voidPtrType);
    }
    else if (varId.getStorageClass() == StorageClass.EXTDEF) {
      commDescId = _globalDecl.declGlobalIdent(XMP.CAF_COMM_PREFIX_ + coarrayName, Xtype.voidPtrType);
    }
    else {
      throw new XMPexception("cannot declare coarray descriptor, '" + coarrayName +  "' has a wrong storage class");
    }

    // call init function
  }
}
