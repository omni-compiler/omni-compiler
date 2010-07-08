package exc.xcalablemp;

import exc.object.*;
import java.util.Vector;

public class XMPtranslateGlobalPragma {
  private XobjectFile		_env;
  private XMPglobalDecl		_globalDecl;
  private XMPobjectTable	_globalObjectTable;

  public XMPtranslateGlobalPragma(XMPglobalDecl globalDecl) {
    _env = globalDecl.getEnv();
    _globalDecl = globalDecl;
    _globalObjectTable = globalDecl.getGlobalObjectTable();
  }

  public void translate(Xobject x) throws XMPexception {
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
        XMP.error(x.getLineNo(), "'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }

  private void translateNodes(Xobject nodesPragma) throws XMPexception {
    LineNo lnObj = nodesPragma.getLineNo();
    XobjList nodesDecl = (XobjList)nodesPragma.getArg(1);

    // check <map-type> := { <undefined> | regular }
    int nodesMapType = 0;
    if (nodesDecl.getArg(0) == null) nodesMapType = XMPnodes.MAP_UNDEFINED;
    else nodesMapType = XMPnodes.MAP_REGULAR;

    // check name collision
    String nodesName = nodesDecl.getArg(1).getString();
    checkObjectNameCollision(lnObj, nodesName);

    // declare nodes desciptor
    Ident nodesDescId = _env.declStaticIdent(XMP.DESC_PREFIX_ + nodesName, Xtype.Pointer(Xtype.voidType));

    // declare nodes object
    int nodesDim = 0;
    for (XobjArgs i = nodesDecl.getArg(2).getArgs(); i != null; i = i.nextArgs()) nodesDim++;
    if ((nodesDim > (XMP.MAX_DIM)) || (nodesDim < 1))
      XMP.error(lnObj, "nodes dimension should be less than " + (XMP.MAX_DIM + 1));

    XMPnodes nodesObject = new XMPnodes(lnObj.lineNo(), nodesName, nodesDim, nodesDescId);
    _globalObjectTable.putObject(nodesObject);

    // create function call
    XobjList nodesArgs = Xcons.List(Xcons.IntConstant(nodesMapType), nodesDescId.getAddr(), Xcons.IntConstant(nodesDim));

    XobjList inheritDecl = (XobjList)nodesDecl.getArg(3);
    String inheritType = null;
    String nodesRefType = null;
    XobjList nodesRef = null;
    XMPnodes nodesRefObject = null;
    switch (inheritDecl.getArg(0).getInt()) {
      case XMPnodes.INHERIT_GLOBAL:
        inheritType = "GLOBAL";
        break;
      case XMPnodes.INHERIT_EXEC:
        inheritType = "EXEC";
        break;
      case XMPnodes.INHERIT_NODES:
        {
          inheritType = "NODES";

          nodesRef = (XobjList)inheritDecl.getArg(1);
          if (nodesRef.getArg(0) == null) {
            nodesRefType = "NUMBER";

            XobjList nodeNumberTriplet = (XobjList)nodesRef.getArg(1);
            // lower
            if (nodeNumberTriplet.getArg(0) == null) nodesArgs.add(Xcons.IntConstant(1));
            else nodesArgs.add(nodeNumberTriplet.getArg(0));
            // upper
            if (nodeNumberTriplet.getArg(1) == null) nodesArgs.add(_globalDecl.getWorldSizeId().Ref());
            else nodesArgs.add(nodeNumberTriplet.getArg(1));
            // stride
            if (nodeNumberTriplet.getArg(2) == null) nodesArgs.add(Xcons.IntConstant(1));
            else nodesArgs.add(nodeNumberTriplet.getArg(2));
          }
          else {
            nodesRefType = "NAMED";

            String nodesRefName = nodesRef.getArg(0).getString();
            nodesRefObject = _globalObjectTable.getNodes(nodesRefName);
            if (nodesRefObject == null)
              XMP.error(lnObj, "cannot find nodes '" + nodesRefName + "'");
            else {
              nodesArgs.add(nodesRefObject.getDescId().Ref());

              int nodesRefDim = nodesRefObject.getDim();
              boolean isDynamicNodesRef = false;
              XobjList subscriptList = (XobjList)nodesRef.getArg(1);
              if (subscriptList == null) {
                for (int nodesRefIndex = 0; nodesRefIndex < nodesRefDim; nodesRefIndex++) {
                  // lower
                  nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
                  // upper
                  Xobject nodesRefSize = nodesRefObject.getUpperAt(nodesRefIndex);
                  if (nodesRefSize == null) isDynamicNodesRef = true;
                  else nodesArgs.add(Xcons.Cast(Xtype.intType, nodesRefSize));
                  // stride
                  nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
                }
              }
              else {
                int nodesRefIndex = 0;
                for (XobjArgs i = subscriptList.getArgs(); i != null; i = i.nextArgs()) {
                  if (nodesRefIndex == nodesRefDim)
                    XMP.error(lnObj, "wrong nodes dimension indicated, too many");

                  XobjList subscriptTriplet = (XobjList)i.getArg();
                  // lower
                  if (subscriptTriplet.getArg(0) == null) nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
                  else nodesArgs.add(Xcons.Cast(Xtype.intType, subscriptTriplet.getArg(0)));
                  // upper
                  if (subscriptTriplet.getArg(1) == null) {
                    Xobject nodesRefSize = nodesRefObject.getUpperAt(nodesRefIndex);
                    if (nodesRefSize == null) isDynamicNodesRef = true;
                    else nodesArgs.add(Xcons.Cast(Xtype.intType, nodesRefSize));
                  }
                  else nodesArgs.add(Xcons.Cast(Xtype.intType, subscriptTriplet.getArg(1)));
                  // stride
                  if (subscriptTriplet.getArg(2) == null) nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
                  else nodesArgs.add(Xcons.Cast(Xtype.intType, subscriptTriplet.getArg(2)));

                  nodesRefIndex++;
                }

                if (nodesRefIndex != nodesRefDim)
                  XMP.error(lnObj, "the number of <nodes-subscript> should be the same with the nodes dimension");
              }

              if (isDynamicNodesRef) nodesArgs.cons(Xcons.IntConstant(1));
              else                   nodesArgs.cons(Xcons.IntConstant(0));
            }
          }
          break;
        }
      default:
        XMP.fatal("cannot create sub node set, unknown operation in nodes directive");
    }

    boolean isDynamic = false;
    for (XobjArgs i = nodesDecl.getArg(2).getArgs(); i != null; i = i.nextArgs()) {
      Xobject nodesSize = i.getArg();
      if (nodesSize == null) isDynamic = true;
      else nodesArgs.add(Xcons.Cast(Xtype.intType, nodesSize));

      nodesObject.addUpper(nodesSize);
    }

    String allocType = null;
    if (isDynamic)	allocType = "DYNAMIC";
    else		allocType = "STATIC";

    if (nodesRef == null)
      _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_init_nodes_" + allocType + "_" + inheritType, nodesArgs);
    else
      _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_init_nodes_" + allocType + "_" + inheritType + "_" + nodesRefType, nodesArgs);
  }

  private void translateTemplate(Xobject templatePragma) throws XMPexception {
    LineNo lnObj = templatePragma.getLineNo();
    XobjList templateDecl = (XobjList)templatePragma.getArg(1);

    // check name collision
    String templateName = templateDecl.getArg(0).getString();
    checkObjectNameCollision(lnObj, templateName);

    // declare template desciptor
    Ident templateDescId = _env.declStaticIdent(XMP.DESC_PREFIX_ + templateName, Xtype.Pointer(Xtype.voidType));

    // declare template object
    int templateDim = 0;
    for (XobjArgs i = templateDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) templateDim++;
    if ((templateDim > (XMP.MAX_DIM)) || (templateDim < 1))
      XMP.error(lnObj, "template dimension should be less than " + (XMP.MAX_DIM + 1));

    XMPtemplate templateObject = new XMPtemplate(lnObj.lineNo(), templateName, templateDim, templateDescId);
    _globalObjectTable.putObject(templateObject);

    // create function call
    boolean templateIsFixed = true;
    XobjList templateArgs = Xcons.List(templateDescId.getAddr(), Xcons.IntConstant(templateDim));
    for (XobjArgs i = templateDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      Xobject templateSpec = i.getArg();
      if (templateSpec == null) {
        templateIsFixed = false;

        templateObject.addLower(null);
        templateObject.addUpper(null);
      }
      else {
        Xobject templateLower = templateSpec.left();
        Xobject templateUpper = templateSpec.right();

        templateArgs.add(Xcons.Cast(Xtype.longlongType, templateLower));
        templateArgs.add(Xcons.Cast(Xtype.longlongType, templateUpper));

        templateObject.addLower(templateLower);
        templateObject.addUpper(templateUpper);
      }
    }

    String fixedSurfix = null;
    if (templateIsFixed) {
      templateObject.setIsFixed();
      fixedSurfix = "FIXED";
    }
    else fixedSurfix = "UNFIXED";

    _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_init_template_" + fixedSurfix, templateArgs);
  }

  private void checkObjectNameCollision(LineNo lnObj, String name) throws XMPexception {
    // check name collision - global variables
    if (_env.findVarIdent(name) != null)
      XMP.error(lnObj, "'" + name + "' is already declared");

    // check name collision - global object table
    if (_globalObjectTable.getObject(name) != null) {
      int ln = _globalObjectTable.getObject(name).getLineNo();
      XMP.error(lnObj, "'" + name + "' is already declared in line." + ln);
    }

    // check name collision - descriptor name
    if (_env.findVarIdent(XMP.DESC_PREFIX_ + name) != null) {
      // FIXME generate unique name
      XMP.error(lnObj, "cannot declare desciptor, '" + XMP.DESC_PREFIX_ + name + "' is already declared");
    }
  }

  private void translateDistribute(Xobject distributePragma) throws XMPexception {
    LineNo lnObj = distributePragma.getLineNo();
    XobjList distDecl = (XobjList)distributePragma.getArg(1);

    // get template object
    String templateName = distDecl.getArg(0).getString();
    XMPtemplate templateObject = _globalObjectTable.getTemplate(templateName);
    if (templateObject == null)
      XMP.error(lnObj, "template '" + templateName + "' is not declared");

    if (templateObject.isDistributed())
      XMP.error(lnObj, "template '" + templateName + "' is already distributed");

    if (!templateObject.isFixed())
      XMP.error(lnObj, "the size of template '" + templateName + "' is not fixed");

    // get nodes object
    String nodesName = distDecl.getArg(2).getString();
    XMPnodes nodesObject = _globalObjectTable.getNodes(nodesName);
    if (nodesObject == null)
      XMP.error(lnObj, "nodes '" + nodesName + "' is not declared");

    templateObject.setOntoNodes(nodesObject);

    // setup chunk constructor
    _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_init_template_chunk",
                                      Xcons.List(templateObject.getDescId().Ref(),
                                                 nodesObject.getDescId().Ref()));

    // create distribute function calls
    int templateDim = templateObject.getDim();
    int templateDimIdx = 0;
    int nodesDim = nodesObject.getDim();
    int nodesDimIdx = 0;
    for (XobjArgs i = distDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      if (templateDimIdx == templateDim)
        XMP.error(lnObj, "wrong template dimension indicated, too many");

      int distManner = i.getArg().getInt();
      // FIXME support cyclic(w), gblock
      switch (distManner) {
        case XMPtemplate.DUPLICATION:
        case XMPtemplate.BLOCK:
        case XMPtemplate.CYCLIC:
          if (nodesDimIdx == nodesDim)
            XMP.error(lnObj, "the number of <dist-format> (except '*') should be the same with the nodes dimension");

          setupDistribution(distManner, templateObject, templateDimIdx, nodesObject, nodesDimIdx);
          nodesDimIdx++;
          break;
        default:
          XMP.fatal("unknown distribute manner");
      }

      templateDimIdx++;
    }

    // check nodes, template dimension
    if (nodesDimIdx != nodesDim)
      XMP.error(lnObj, "the number of <dist-format> (except '*') should be the same with the nodes dimension");

    if (templateDimIdx != templateDim)
      XMP.error(lnObj, "wrong template dimension indicated, too few");

    // set distributed
    templateObject.setIsDistributed();
  }

  private void setupDistribution(int distManner,
                                 XMPtemplate templateObject, int templateDimIdx,
                                 XMPnodes nodesObject,       int nodesDimIdx) {
    String distMannerName = null;
    switch (distManner) {
      case XMPtemplate.DUPLICATION:
        distMannerName = "DUPLICATION";
        break;
      case XMPtemplate.BLOCK:
        distMannerName = "BLOCK";
        break;
      case XMPtemplate.CYCLIC:
        distMannerName = "CYCLIC";
        break;
      default:
        XMP.fatal("unknown distribute manner");
    }

    _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_dist_template_" + distMannerName,
                                      Xcons.List(templateObject.getDescId().Ref(),
                                                 Xcons.IntConstant(templateDimIdx),
                                                 nodesObject.getDescId().Ref(),
                                                 Xcons.IntConstant(nodesDimIdx)));
    templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
    templateObject.setDistMannerAt(distManner, templateDimIdx);
  }

  private void translateAlign(Xobject alignPragma) throws XMPexception {
    LineNo lnObj = alignPragma.getLineNo();
    Xobject alignDecl = alignPragma.getArg(1);

    // get array information
    String arrayName = alignDecl.getArg(0).getString();
    if (_globalObjectTable.getAlignedArray(arrayName) != null)
      XMP.error(lnObj, "array '" + arrayName + "' is already aligned");

    Ident arrayId = _env.findVarIdent(arrayName);
    if (arrayId == null)
      XMP.error(lnObj, "array '" + arrayName + "' is not declared");

    Xtype arrayType = arrayId.Type();
    if (arrayType.getKind() != Xtype.ARRAY)
      XMP.error(lnObj, arrayName + " is not an array");

    Xtype arrayElmtType = arrayType.getArrayElementType();

    // get template information
    String templateName = alignDecl.getArg(2).getString();
    XMPtemplate templateObj = _globalObjectTable.getTemplate(templateName);
    if (templateObj == null)
      XMP.error(lnObj, "template '" + templateName + "' is not declared");

    if (!(templateObj.isDistributed()))
      XMP.error(lnObj, "template '" + templateName + "' is not distributed");

    int templateDim = templateObj.getDim();

    // declare array address pointer
    Ident arrayAddrId = null;
    if( arrayId.getStorageClass() == StorageClass.EXTERN)
      arrayAddrId = _env.declExternIdent(XMP.ADDR_PREFIX_ + arrayName,
                                         Xtype.Pointer(arrayElmtType));
    else if (arrayId.getStorageClass() == StorageClass.STATIC)
      arrayAddrId = _env.declStaticIdent(XMP.ADDR_PREFIX_ + arrayName,
                                         Xtype.Pointer(arrayElmtType));
    else if(arrayId.getStorageClass() == StorageClass.EXTDEF)
      arrayAddrId = _env.declGlobalIdent(XMP.ADDR_PREFIX_ + arrayName,
                                         Xtype.Pointer(arrayElmtType));
    else
      XMP.error(lnObj, "cannot align array '" + arrayName +  ", wrong storage class");

    // declare array descriptor
    Ident arrayDescId = _env.declStaticIdent(XMP.DESC_PREFIX_ + arrayName,
                                             Xtype.Pointer(Xtype.voidType));

    int arrayDim = arrayType.getNumDimensions();
    if ((arrayDim > (XMP.MAX_DIM)) || (arrayDim < 1))
      XMP.error(lnObj, "array dimension should be less than " + (XMP.MAX_DIM + 1));

    XobjList initArrayDescFuncArgs = Xcons.List(arrayDescId.getAddr(),
                                                templateObj.getDescId().Ref(),
                                                Xcons.IntConstant(arrayDim));

    Vector<Long> arraySizeVector = new Vector<Long>(arrayDim);
    Vector<Ident> gtolAccIdVector = new Vector<Ident>(arrayDim);
    for (int i = 0; i < arrayDim; i++, arrayType = arrayType.getRef()) {
      long dimSize = arrayType.getArraySize();
      if(dimSize == 0)
        XMP.error(lnObj, "array size cannot be omitted");

      arraySizeVector.add(new Long(dimSize));
      initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));

      Ident gtolAccId = _env.declStaticIdent(XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i,
                                             Xtype.unsignedlonglongType);
      gtolAccIdVector.add(gtolAccId);
    }

    _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_init_array_desc", initArrayDescFuncArgs);

    XMPalignedArray alignedArray = new XMPalignedArray(lnObj, arrayName, arrayElmtType, arrayDim,
                                                       arraySizeVector, gtolAccIdVector, arrayDescId, arrayAddrId);
    _globalObjectTable.putAlignedArray(alignedArray);

    // check <align-source> list, <align-subscrip> list
    XobjList alignSourceList = (XobjList)alignDecl.getArg(1);
    XobjList alignSubscriptList = (XobjList)alignDecl.getArg(3);
    XobjList alignSubscriptVarList = (XobjList)alignSubscriptList.left();
    XobjList alignSubscriptExprList = (XobjList)alignSubscriptList.right();

    // check <align-source> list
    if (XMPutil.countElmts(alignSourceList) != arrayDim)
      XMP.error(lnObj, "the number of <align-source>s is not the same with array dimension");

    // check <align-subscript> list
    if (XMPutil.countElmts(alignSubscriptVarList) != templateDim)
      XMP.error(lnObj, "the number of <align-subscript>s is not the same with template dimension");

    // check ':' source/subscript
    if (XMPutil.countElmts(alignSourceList, XMP.COLON) !=
        XMPutil.countElmts(alignSubscriptVarList, XMP.COLON))
      XMP.error(lnObj, "the number of ':' in <align-source> list is not the same with <align-subscript> list");

    // create align function calls
    int alignSourceIndex = 0;
    for (XobjArgs i = alignSourceList.getArgs(); i != null; i = i.nextArgs()) {
      Xobject alignSourceObj = i.getArg();
      if (alignSourceObj.Opcode() == Xcode.INT_CONSTANT) {
        int alignSource = alignSourceObj.getInt();
        if (alignSource == XMPalignedArray.NO_ALIGN) continue;
        else if (alignSource == XMPalignedArray.SIMPLE_ALIGN) {
          if (!XMPutil.hasElmt(alignSubscriptVarList, XMPalignedArray.SIMPLE_ALIGN))
            XMP.error(lnObj, "cannot find ':' in <align-subscript> list");

          int alignSubscriptIndex = XMPutil.getLastIndex(alignSubscriptVarList, XMPalignedArray.SIMPLE_ALIGN);
          alignSubscriptVarList.setArg(alignSubscriptIndex, null);

          declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex, null);
        }
      }
      else if (alignSourceObj.Opcode() == Xcode.STRING) {
        String alignSource = alignSourceObj.getString();
        if (XMPutil.countElmts(alignSourceList, alignSource) != 1)
          XMP.error(lnObj, "multiple '" + alignSource + "' indicated in <align-source> list");

        if (!XMPutil.hasElmt(alignSubscriptVarList, alignSource)) continue;

        if (XMPutil.countElmts(alignSubscriptVarList, alignSource) != 1)
          XMP.error(lnObj, "multiple '" + alignSource + "' indicated in <align-subscript> list");

        int alignSubscriptIndex = XMPutil.getFirstIndex(alignSubscriptVarList, alignSource);
        declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
                      alignSubscriptExprList.getArg(alignSubscriptIndex));
      }

      alignSourceIndex++;
    }
  }

  private void declAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                             XMPtemplate templateObj, int alignSubscriptIndex,
                             Xobject alignSubscriptExpr) throws XMPexception {
    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex),
                                        templateObj.getDescId().Ref(),
                                        Xcons.IntConstant(alignSubscriptIndex));

    if (alignSubscriptExpr == null) alignFuncArgs.add(Xcons.IntConstant(0));
    else alignFuncArgs.add(alignSubscriptExpr);

    int distManner = templateObj.getDistMannerAt(alignSubscriptIndex);
    alignedArray.setDistMannerAt(distManner, alignSourceIndex);

    switch (distManner) {
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
        {
          Ident gtolTemp0Id = _env.declStaticIdent(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
                                                   Xtype.intType);
          alignedArray.setGtolTemp0IdAt(gtolTemp0Id, alignSourceIndex);
          alignFuncArgs.add(gtolTemp0Id.getAddr());

          break;
        }
      default:
        XMP.fatal("unknown distribute manner");
    }

    _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
                                      alignFuncArgs);
  }

  // FIXME incomplete, not checked
  private void translateShadow(Xobject shadowPragma) throws XMPexception {
    LineNo lnObj = shadowPragma.getLineNo();
    XobjList shadowDecl = (XobjList)shadowPragma.getArg(1);

    // find aligned array
    String arrayName = shadowDecl.getArg(0).getString();
    XMPalignedArray alignedArray = _globalObjectTable.getAlignedArray(arrayName);
    if (alignedArray == null)
      XMP.error(lnObj, "the aligned array '" + arrayName + "' is not found");

    if (alignedArray.hasShadow())
      XMP.error(lnObj, "the aligned array '" + arrayName + "' has shadow already");

    // init shadow
    XobjList shadowFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
    int arrayIndex = 0;
    int arrayDim = alignedArray.getDim();
    for (XobjArgs i = shadowDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      if (arrayIndex == arrayDim)
        XMP.error(lnObj, "wrong shadow dimension indicated, too many");

      XobjList shadowObj = (XobjList)i.getArg();
      XobjInt shadowType = (XobjInt)shadowObj.getArg(0);
      XobjList shadowBody = (XobjList)shadowObj.getArg(1);
      switch (shadowType.getInt()) {
        case XMPshadow.SHADOW_NONE:
          {
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_NONE)));

            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_NONE, null, null), arrayIndex);
            break;
          }
        case XMPshadow.SHADOW_NORMAL:
          {
            if (alignedArray.getDistMannerAt(arrayIndex) == XMPalignedArray.NO_ALIGN)
              XMP.error(lnObj, "indicated dimension is not distributed");

            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_NORMAL)));
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, shadowBody.left()));
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, shadowBody.right()));

            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_NORMAL, shadowBody.left(), shadowBody.right()), arrayIndex);
            break;
          }
        case XMPshadow.SHADOW_FULL:
          {
            if (alignedArray.getDistMannerAt(arrayIndex) == XMPalignedArray.NO_ALIGN)
              XMP.error(lnObj, "indicated dimension is not distributed");

            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_FULL)));

            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_FULL, null, null), arrayIndex);
            break;
          }
        default:
          XMP.error(lnObj, "unknown shadow type");
      }

      arrayIndex++;
    }

    if (arrayIndex != arrayDim)
      XMP.error(lnObj, "the number of <nodes/template-subscript> should be the same with the dimension");

    _globalDecl.addGlobalInitFuncCall("_XCALABLEMP_init_shadow", shadowFuncArgs);

    // set shadow flag
    alignedArray.setHasShadow();
  }
}
