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
        { translateNodes(x);		break; }
      case TEMPLATE:
        { translateTemplate(x);		break; }
      case DISTRIBUTE:
        { translateDistribute(x);	break; }
      case ALIGN:
        { translateAlign(x);		break; }
      case SHADOW:
        { translateShadow(x);		break; }
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
    Xobject alignDecl = alignPragma.getArg(1);

    // get array information
    String arrayName = alignDecl.getArg(0).getString();
    if (_globalDecl.getXMPalignedArray(arrayName) != null) {
      throw new XMPexception("array '" + arrayName + "' is already aligned");
    }

    Ident arrayId = _globalDecl.findVarIdent(arrayName);
    if (arrayId == null) {
      throw new XMPexception("array '" + arrayName + "' is not declared");
    }

    Xtype arrayType = arrayId.Type();
    if (arrayType.getKind() != Xtype.ARRAY) {
      throw new XMPexception(arrayName + " is not an array");
    }

    Xtype arrayElmtType = arrayType.getArrayElementType();
    Xobject arrayElmtTypeRef = null;
    if (arrayElmtType.getKind() == Xtype.BASIC) {
      arrayElmtTypeRef = XMP.createBasicTypeConstantObj(arrayElmtType);
    }
    else {
      arrayElmtTypeRef = Xcons.IntConstant(XMP.NONBASIC_TYPE);
    }

    // get template information
    String templateName = alignDecl.getArg(2).getString();
    XMPtemplate templateObj = _globalDecl.getXMPtemplate(templateName);
    if (templateObj == null) {
      throw new XMPexception("template '" + templateName + "' is not declared");
    }

    if (!templateObj.isFixed()) {
      throw new XMPexception("template '" + templateName + "' is not fixed");
    }

    if (!(templateObj.isDistributed())) {
      throw new XMPexception("template '" + templateName + "' is not distributed");
    }

    int templateDim = templateObj.getDim();

    // declare array address pointer
    Ident arrayAddrId = null;
    if (arrayId.getStorageClass() == StorageClass.EXTERN) {
      arrayAddrId = _globalDecl.declExternIdent(XMP.ADDR_PREFIX_ + arrayName,
                                         Xtype.Pointer(arrayElmtType));
    }
    else if (arrayId.getStorageClass() == StorageClass.STATIC) {
      arrayAddrId = _globalDecl.declStaticIdent(XMP.ADDR_PREFIX_ + arrayName,
                                         Xtype.Pointer(arrayElmtType));
    }
    else if (arrayId.getStorageClass() == StorageClass.EXTDEF) {
      arrayAddrId = _globalDecl.declGlobalIdent(XMP.ADDR_PREFIX_ + arrayName,
                                         Xtype.Pointer(arrayElmtType));
    }
    else {
      throw new XMPexception("cannot align array '" + arrayName +  ", wrong storage class");
    }

    // declare array descriptor
    Ident arrayDescId = _globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + arrayName,
                                             Xtype.voidPtrType);

    int arrayDim = arrayType.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM) {
      throw new XMPexception("array dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    XobjList initArrayDescFuncArgs = Xcons.List(arrayDescId.getAddr(),
                                                templateObj.getDescId().Ref(),
                                                Xcons.IntConstant(arrayDim),
                                                arrayElmtTypeRef,
                                                Xcons.SizeOf(arrayElmtType));

    Vector<Long> arraySizeVector = new Vector<Long>(arrayDim);
    Vector<Ident> accIdVector = new Vector<Ident>(arrayDim);
    for (int i = 0; i < arrayDim; i++, arrayType = arrayType.getRef()) {
      long dimSize = arrayType.getArraySize();
      if (dimSize == 0) {
        throw new XMPexception("array size cannot be omitted");
      }
      else if (dimSize == -1) {
        // FIXME possible error in global scope???
        throw new XMPexception("array size should be fixed");
      }

      arraySizeVector.add(new Long(dimSize));
      initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));

      Ident accId = _globalDecl.declStaticIdent(XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i,
                                         Xtype.unsignedlonglongType);
      accIdVector.add(accId);
    }

    _globalDecl.addGlobalInitFuncCall("_XMP_init_array_desc", initArrayDescFuncArgs);

    XMPalignedArray alignedArray = new XMPalignedArray(arrayName, arrayElmtType, arrayDim,
                                                       arraySizeVector, accIdVector,
                                                       arrayId, arrayDescId, arrayAddrId,
                                                       templateObj);
    _globalDecl.putXMPalignedArray(alignedArray);

    // check <align-source> list, <align-subscrip> list
    XobjList alignSourceList = (XobjList)alignDecl.getArg(1);
    XobjList alignSubscriptList = (XobjList)alignDecl.getArg(3);
    XobjList alignSubscriptVarList = (XobjList)alignSubscriptList.left();
    XobjList alignSubscriptExprList = (XobjList)alignSubscriptList.right();

    // check <align-source> list
    if (XMPutil.countElmts(alignSourceList) != arrayDim) {
      throw new XMPexception("the number of <align-source>s is not the same with array dimension");
    }
    else if (XMPutil.countElmts(alignSourceList, XMP.ASTERISK) == arrayDim) {
      throw new XMPexception("array " + arrayName + " is not aligned on any dimension");
    }

    // check <align-subscript> list
    if (XMPutil.countElmts(alignSubscriptVarList) != templateDim) {
      throw new XMPexception("the number of <align-subscript>s is not the same with template dimension");
    }

    // check ':' source/subscript
    if (XMPutil.countElmts(alignSourceList, XMP.COLON) !=
        XMPutil.countElmts(alignSubscriptVarList, XMP.COLON)) {
      throw new XMPexception("the number of ':' in <align-source> list is not the same with <align-subscript> list");
    }

    // create align function calls
    int alignSourceIndex = 0;
    for (XobjArgs i = alignSourceList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSource = i.getArg().getString();

      if (alignSource.equals(XMP.ASTERISK)) {
        declNotAlignFunc(alignedArray, alignSourceIndex);
      }
      else if (alignSource.equals(XMP.COLON)) {
        if (!XMPutil.hasElmt(alignSubscriptVarList, XMP.COLON))
          throw new XMPexception("cannot find ':' in <align-subscript> list");

        int alignSubscriptIndex = XMPutil.getLastIndex(alignSubscriptVarList, XMP.COLON);
        alignSubscriptVarList.setArg(alignSubscriptIndex, null);

        declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex, null);
      }
      else {
        if (XMPutil.countElmts(alignSourceList, alignSource) != 1)
          throw new XMPexception("multiple '" + alignSource + "' indicated in <align-source> list");

        if (XMPutil.hasElmt(alignSubscriptVarList, alignSource)) {
          if (XMPutil.countElmts(alignSubscriptVarList, alignSource) != 1)
            throw new XMPexception("multiple '" + alignSource + "' indicated in <align-subscript> list");

          int alignSubscriptIndex = XMPutil.getFirstIndex(alignSubscriptVarList, alignSource);
          declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
                        alignSubscriptExprList.getArg(alignSubscriptIndex));
        }
        else
          throw new XMPexception("cannot find '" + alignSource + "' in <align-subscript> list");
      }

      alignSourceIndex++;
    }

    // check alignSubscriptVarList
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();

      if (alignSubscript.equals(XMP.ASTERISK) || alignSubscript.equals(XMP.COLON)) break;

      if (XMPutil.hasElmt(alignSourceList, alignSubscript)) {
        if (XMPutil.countElmts(alignSourceList, alignSubscript) != 1)
          throw new XMPexception("no/multiple '" + alignSubscript + "' indicated in <align-source> list");
      }
      else
        throw new XMPexception("cannot find '" + alignSubscript + "' in <align-source> list");
    }

    // init array communicator
    XobjList initArrayCommFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();

      if (alignSubscript.equals(XMP.ASTERISK)) initArrayCommFuncArgs.add(Xcons.IntConstant(1));
      else                                     initArrayCommFuncArgs.add(Xcons.IntConstant(0));
    }

    _globalDecl.addGlobalInitFuncCall("_XMP_init_array_comm", initArrayCommFuncArgs);
  }

  private void declNotAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex) throws XMPexception {
    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex));

    alignedArray.setAlignMannerAt(XMPalignedArray.NOT_ALIGNED, alignSourceIndex);

    _globalDecl.addGlobalInitFuncCall("_XMP_align_array_NOT_ALIGNED", alignFuncArgs);
  }

  private void declAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                             XMPtemplate templateObj, int alignSubscriptIndex,
                             Xobject alignSubscriptExpr) throws XMPexception {
    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex),
                                        Xcons.IntConstant(alignSubscriptIndex));

    if (alignSubscriptExpr == null) alignFuncArgs.add(Xcons.IntConstant(0));
    else alignFuncArgs.add(alignSubscriptExpr);

    int distManner = templateObj.getDistMannerAt(alignSubscriptIndex);
    alignedArray.setAlignMannerAt(XMPalignedArray.convertDistMannerToAlignManner(distManner), alignSourceIndex);

    alignedArray.setAlignSubscriptIndexAt(alignSubscriptIndex, alignSourceIndex);
    alignedArray.setAlignSubscriptExprAt(alignSubscriptExpr, alignSourceIndex);

    switch (distManner) {
      case XMPtemplate.DUPLICATION: // FIXME how implement???
        break;
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
        {
          Ident gtolTemp0Id = _globalDecl.declStaticIdent(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
                                                   Xtype.intType);
          alignedArray.setGtolTemp0IdAt(gtolTemp0Id, alignSourceIndex);
          alignFuncArgs.add(gtolTemp0Id.getAddr());

          break;
        }
      default:
        throw new XMPexception("unknown distribute manner");
    }

    _globalDecl.addGlobalInitFuncCall("_XMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
                                      alignFuncArgs);
  }

  private void translateShadow(Xobject shadowPragma) throws XMPexception {
    XMPshadow.translateShadow((XobjList)shadowPragma.getArg(1), _globalDecl, false, null);
  }
}
