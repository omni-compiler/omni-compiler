/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.util.*;

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
      case LOCAL_ALIAS:
        translateLocalAlias(x);
        break;
      default:
        throw new XMPexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }

  private void translateNodes(Xobject nodesPragma) throws XMPexception {
    XobjList nodesDecl = (XobjList)nodesPragma.getArg(1);
    XobjList nodesNameList = (XobjList)nodesDecl.getArg(0);
    XobjList nodesDeclCopy = (XobjList)nodesDecl.copy();

    Iterator<Xobject> iter = nodesNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      nodesDeclCopy.setArg(0, x);
      XMPnodes.translateNodes(nodesDeclCopy, _globalDecl, false, null);
    }
  }

  private void translateTemplate(Xobject templatePragma) throws XMPexception {
    XobjList templateDecl = (XobjList)templatePragma.getArg(1);
    XobjList templateNameList = (XobjList)templateDecl.getArg(0);
    XobjList templateDeclCopy = (XobjList)templateDecl.copy();

    Iterator<Xobject> iter = templateNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      templateDeclCopy.setArg(0, x);
      XMPtemplate.translateTemplate(templateDeclCopy, _globalDecl, false, null);
    }
  }

  private void translateDistribute(Xobject distributePragma) throws XMPexception {
    XobjList distributeDecl = (XobjList)distributePragma.getArg(1);
    XobjList distributeNameList = (XobjList)distributeDecl.getArg(0);
    XobjList distributeDeclCopy = (XobjList)distributeDecl.copy();

    Iterator<Xobject> iter = distributeNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      distributeDeclCopy.setArg(0, x);
      XMPtemplate.translateDistribute(distributeDeclCopy, _globalDecl, false, null);
    }
  }

  private void translateAlign(Xobject alignPragma) throws XMPexception {
    XobjList alignDecl = (XobjList)alignPragma.getArg(1);
    XobjList alignNameList = (XobjList)alignDecl.getArg(0);
    XobjList alignDeclCopy = (XobjList)alignDecl.copy();

    Iterator<Xobject> iter = alignNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      alignDeclCopy.setArg(0, x);
      XMPalignedArray.translateAlign(alignDeclCopy, _globalDecl, false, null);
    }
  }

  private void translateLocalAlias(Xobject localAliasPragma) throws XMPexception {
    XMPalignedArray.translateLocalAlias((XobjList)localAliasPragma.getArg(1), _globalDecl, false, null);
  }

  private void translateShadow(Xobject shadowPragma) throws XMPexception {
    XobjList shadowDecl = (XobjList)shadowPragma.getArg(1);
    XobjList shadowNameList = (XobjList)shadowDecl.getArg(0);
    XobjList shadowDeclCopy = (XobjList)shadowDecl.copy();

    Iterator<Xobject> iter = shadowNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      shadowDeclCopy.setArg(0, x);
      XMPshadow.translateShadow(shadowDeclCopy, _globalDecl, false, null);
    }
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

    boolean isArray = false;
    int varDim = 0;
    Xtype elmtType = null;
    Xtype varType = varId.Type();
    Xobject varAddr = null;
    if (varType.getKind() == Xtype.ARRAY) {
      isArray = true;
      varDim = varType.getNumDimensions();
      elmtType = varType.getArrayElementType();
      varAddr = varId.Ref();
    } else {
      varDim = 1;
      elmtType = varType;
      varAddr = varId.getAddr();
    }

    Xobject elmtTypeRef = null;
    if (elmtType.getKind() == Xtype.BASIC) {
      elmtTypeRef = XMP.createBasicTypeConstantObj(elmtType);
    } else {
      elmtTypeRef = Xcons.IntConstant(XMP.NONBASIC_TYPE);
    }

    // init descriptor
    Ident descId = _globalDecl.declStaticIdent(XMP.COARRAY_DESC_PREFIX_ + coarrayName, Xtype.voidPtrType);
    XobjList initDescFuncArgs = Xcons.List(descId.getAddr(), varAddr, elmtTypeRef, Xcons.SizeOf(elmtType));

    String initDescFuncName = null;
    if (coarrayDecl.getArg(1) == null) {
      initDescFuncName = new String("_XMP_init_coarray_DYNAMIC");
    } else {
      initDescFuncName = new String("_XMP_init_coarray_STATIC");
      initDescFuncArgs.add(coarrayDecl.getArg(1));
    }

    initDescFuncArgs.add(Xcons.IntConstant(varDim));

    Vector<Long> sizeVector = new Vector<Long>(varDim);
    if (isArray) {
      for (int i = 0; i < varDim; i++, varType = varType.getRef()) {
        long dimSize = varType.getArraySize();
        if (dimSize == 0) {
          throw new XMPexception("array size cannot be omitted");
        } else if (dimSize == -1) {
          // FIXME possible error in global scope???
          throw new XMPexception("array size should be fixed");
        }

        sizeVector.add(new Long(dimSize));
        initDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));
      }
    } else {
      sizeVector.add(new Long(1));
      initDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
    }

    XMPcoarray coarrayEntry = new XMPcoarray(coarrayName, elmtType, varDim, sizeVector, varAddr, varId, descId);

    _globalDecl.putXMPcoarray(coarrayEntry);
    _globalDecl.addGlobalInitFuncCall(initDescFuncName, initDescFuncArgs);

    // call finalize function
    _globalDecl.addGlobalFinalizeFuncCall("_XMP_finalize_coarray", Xcons.List(descId.Ref()));
  }
}
