package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.util.Vector;
import java.util.Iterator;

public class XMPtranslateLocalPragma {
  private XMPglobalDecl		_globalDecl;
  private XobjectFile		_env;
  private XMPobjectTable	_globalObjectTable;

  public XMPtranslateLocalPragma(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
    _env = globalDecl.getEnv();
    _globalObjectTable = globalDecl.getGlobalObjectTable();
  }

  public void translate(FuncDefBlock def) {
    FunctionBlock fb = def.getBlock();

    BlockIterator i = new topdownBlockIterator(fb);
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() ==  Xcode.XMP_PRAGMA) {
        PragmaBlock pb = (PragmaBlock)b;

        try {
          translatePragma(pb);
        } catch (XMPexception e) {
          XMP.error(pb.getLineNo(), e.getMessage());
        }
      }
    }

    def.Finalize();
  }

  private void translatePragma(PragmaBlock pb) throws XMPexception {
    String pragmaName = pb.getPragma();

    switch (XMPpragma.valueOf(pragmaName)) {
      case NODES:
        { translateNodes(pb);		break; }
      case TEMPLATE:
        { translateTemplate(pb);	break; }
      case DISTRIBUTE:
        { translateDistribute(pb);	break; }
      case ALIGN:
        { translateAlign(pb);		break; }
      case SHADOW:
        { translateShadow(pb);		break; }
      case TASK:
        { translateTask(pb);		break; }
      case TASKS:
        { translateTasks(pb);		break; }
      case LOOP:
        { translateLoop(pb);		break; }
      case REFLECT:
        { translateReflect(pb);		break; }
      case BARRIER:
        { translateBarrier(pb);		break; }
      case REDUCTION:
        { translateReduction(pb);	break; }
      case BCAST:
        { translateBcast(pb);		break; }
      case GMOVE:
        { translateGmove(pb);           break; }
      default:
        throw new XMPexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }

  private void translateNodes(PragmaBlock pb) throws XMPexception {
    // check location
    checkDeclPragmaLocation(pb);

    // start translation
    XobjList nodesDecl = (XobjList)pb.getClauses();
    FunctionBlock functionBlock = XMPlocalDecl.findParentFunctionBlock(pb);
    BlockList funcBlockList = functionBlock.getBody();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // check <map-type> := { <undefined> | regular }
    int nodesMapType = 0;
    if (nodesDecl.getArg(0) == null) nodesMapType = XMPnodes.MAP_UNDEFINED;
    else nodesMapType = XMPnodes.MAP_REGULAR;

    // check name collision
    String nodesName = nodesDecl.getArg(1).getString();
    checkObjectNameCollision(nodesName, funcBlockList, localObjectTable);

    // declare nodes desciptor
    Ident nodesDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + nodesName, pb);

    // declare nodes object
    int nodesDim = 0;
    for (XobjArgs i = nodesDecl.getArg(2).getArgs(); i != null; i = i.nextArgs()) nodesDim++;
    if ((nodesDim > (XMP.MAX_DIM)) || (nodesDim < 1))
      throw new XMPexception("nodes dimension should be less than " + (XMP.MAX_DIM + 1));

    XMPnodes nodesObject = new XMPnodes(nodesName, nodesDim, nodesDescId);
    localObjectTable.putObject(nodesObject);

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
            nodesRefObject = getXMPnodes(nodesRefName, localObjectTable);
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
                  throw new XMPexception("wrong nodes dimension indicated, too many");

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
                throw new XMPexception("the number of <nodes-subscript> should be the same with the nodes dimension");
            }

            if (isDynamicNodesRef) nodesArgs.cons(Xcons.IntConstant(1));
            else                   nodesArgs.cons(Xcons.IntConstant(0));
          }
          break;
        }
      default:
        throw new XMPexception("cannot create sub node set, unknown operation in nodes directive");
    }

    boolean isDynamic = false;
    for (XobjArgs i = nodesDecl.getArg(2).getArgs(); i != null; i = i.nextArgs()) {
      Xobject nodesSize = i.getArg();
      if (nodesSize == null) isDynamic = true;
      else nodesArgs.add(Xcons.Cast(Xtype.intType, nodesSize));

      nodesObject.addUpper(nodesSize);
    }

    String allocType = null;
    if (isDynamic)      allocType = "DYNAMIC";
    else                allocType = "STATIC";

    // add constructor call
    if (nodesRef == null)
      XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_nodes_" + allocType + "_" + inheritType, nodesArgs, pb, _globalDecl);
    else
      XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_nodes_" + allocType + "_" + inheritType + "_" + nodesRefType,
                                      nodesArgs, pb, _globalDecl);

    // insert destructor call
    XMPlocalDecl.insertDestructorCall("_XCALABLEMP_finalize_nodes", Xcons.List(nodesDescId.Ref()), pb, _globalDecl);
  }

  private void translateTemplate(PragmaBlock pb) throws XMPexception {
    // check location
    checkDeclPragmaLocation(pb);

    // start translation
    XobjList templateDecl = (XobjList)pb.getClauses();
    FunctionBlock functionBlock = XMPlocalDecl.findParentFunctionBlock(pb);
    BlockList funcBlockList = functionBlock.getBody();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // check name collision - parameters
    String templateName = templateDecl.getArg(0).getString();
    checkObjectNameCollision(templateName, funcBlockList, localObjectTable);

    // declare template desciptor
    Ident templateDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + templateName, pb);

    // declare template object
    int templateDim = 0;
    for (XobjArgs i = templateDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) templateDim++;
    if ((templateDim > (XMP.MAX_DIM)) || (templateDim < 1))
      throw new XMPexception("template dimension should be less than " + (XMP.MAX_DIM + 1));

    XMPtemplate templateObject = new XMPtemplate(templateName, templateDim, templateDescId);
    localObjectTable.putObject(templateObject);

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

    // add constructor call
    XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_template_" + fixedSurfix, templateArgs, pb, _globalDecl);

    // insert destructor call
    XMPlocalDecl.insertDestructorCall("_XCALABLEMP_finalize_template", Xcons.List(templateDescId.Ref()), pb, _globalDecl);
  }

  private void checkObjectNameCollision(String name, BlockList scopeBL, XMPobjectTable objectTable) throws XMPexception {
    // check name collision - parameters
    if (scopeBL.findLocalIdent(name) != null)
      throw new XMPexception("'" + name + "' is already declared");

    // check name collision - local object table
    if (objectTable.getObject(name) != null)
      throw new XMPexception("'" + name + "' is already declared");

    // check name collision - descriptor name
    if (scopeBL.findLocalIdent(XMP.DESC_PREFIX_ + name) != null) {
      // FIXME generate unique name
      throw new XMPexception("cannot declare template desciptor, '" + XMP.DESC_PREFIX_ + name + "' is already declared");
    }
  }

  private void translateDistribute(PragmaBlock pb) throws XMPexception {
    // check location
    checkDeclPragmaLocation(pb);

    // start translation
    XobjList distDecl = (XobjList)pb.getClauses();
    FunctionBlock functionBlock = XMPlocalDecl.findParentFunctionBlock(pb);
    BlockList funcBlockList = functionBlock.getBody();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // get template object
    String templateName = distDecl.getArg(0).getString();
    XMPtemplate templateObject = localObjectTable.getTemplate(templateName);
    if (templateObject == null) {
      templateObject = _globalObjectTable.getTemplate(templateName);
      if (templateObject == null)
        throw new XMPexception("template '" + templateName + "' is not declared");
      else
        throw new XMPexception("global template cannot be distributed in local scope");
    }

    if (templateObject.isDistributed())
      throw new XMPexception("template '" + templateName + "' is already distributed");

    if (!templateObject.isFixed())
      throw new XMPexception("the size of template '" + templateName + "' is not fixed");

    // get nodes object
    String nodesName = distDecl.getArg(2).getString();
    XMPnodes nodesObject = getXMPnodes(nodesName, localObjectTable);

    templateObject.setOntoNodes(nodesObject);

    // setup chunk constructor & destructor
    XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_template_chunk",
                                    Xcons.List(templateObject.getDescId().Ref(),
                                               nodesObject.getDescId().Ref()),
                                    pb, _globalDecl);

    // create distribute function calls
    int templateDim = templateObject.getDim();
    int templateDimIdx = 0;
    int nodesDim = nodesObject.getDim();
    int nodesDimIdx = 0;
    for (XobjArgs i = distDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      if (templateDimIdx == templateDim)
        throw new XMPexception("wrong template dimension indicated, too many");

      int distManner = i.getArg().getInt();
      // FIXME support cyclic(w), gblock
      switch (distManner) {
        case XMPtemplate.DUPLICATION:
          {
            setupDistribution(pb, distManner, templateObject, templateDimIdx, -1);
            break;
          }
        case XMPtemplate.BLOCK:
        case XMPtemplate.CYCLIC:
          {
            if (nodesDimIdx == nodesDim)
              throw new XMPexception("the number of <dist-format> (except '*') should be the same with the nodes dimension");

            setupDistribution(pb, distManner, templateObject, templateDimIdx, nodesDimIdx);
            nodesDimIdx++;
            break;
          }
        default:
          throw new XMPexception("unknown distribute manner");
      }

      templateDimIdx++;
    }

    // check nodes, template dimension
    if (nodesDimIdx != nodesDim)
      throw new XMPexception("the number of <dist-format> (except '*') should be the same with the nodes dimension");

    if (templateDimIdx != templateDim)
      throw new XMPexception("wrong template dimension indicated, too few");

    // set distributed
    templateObject.setIsDistributed();
  }

  private void setupDistribution(PragmaBlock pb, int distManner, XMPtemplate templateObject,
                                 int templateDimIdx, int nodesDimIdx) throws XMPexception {
    XobjList funcArgs = null;
    String distMannerName = null;
    switch (distManner) {
      case XMPtemplate.DUPLICATION:
        {
          distMannerName = "DUPLICATION";
          funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                Xcons.IntConstant(templateDimIdx));
          break;
        }
      case XMPtemplate.BLOCK:
        {
          distMannerName = "BLOCK";
          funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                Xcons.IntConstant(templateDimIdx),
                                Xcons.IntConstant(nodesDimIdx));
          templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
          break;
        }
      case XMPtemplate.CYCLIC:
        {
          distMannerName = "CYCLIC";
          funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                Xcons.IntConstant(templateDimIdx),
                                Xcons.IntConstant(nodesDimIdx));
          templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
          break;
        }
      default:
        throw new XMPexception("unknown distribute manner");
    }

    XMPlocalDecl.addConstructorCall("_XCALABLEMP_dist_template_" + distMannerName, funcArgs, pb, _globalDecl);
    templateObject.setDistMannerAt(distManner, templateDimIdx);
  }

  private void translateAlign(PragmaBlock pb) throws XMPexception {
    // check location
    checkDeclPragmaLocation(pb);

    // start translation
    XobjList alignDecl = (XobjList)pb.getClauses();
    FunctionBlock functionBlock = XMPlocalDecl.findParentFunctionBlock(pb);
    BlockList funcBlockList = functionBlock.getBody();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // get array information
    String arrayName = alignDecl.getArg(0).getString();
    if (localObjectTable.getAlignedArray(arrayName) != null)
      throw new XMPexception("array '" + arrayName + "' is already aligned");

    Ident arrayId = funcBlockList.findLocalIdent(arrayName);
    if (arrayId == null)
      throw new XMPexception("array '" + arrayName + "' is not declared");

    if (arrayId.getStorageClass() != StorageClass.PARAM)
      throw new XMPexception("array '" + arrayName + "' is not a parameter");

    Xtype arrayType = arrayId.Type();
    if (arrayType.getKind() != Xtype.ARRAY)
      throw new XMPexception(arrayName + " is not an array");

    Xtype arrayElmtType = arrayType.getArrayElementType();

    // get template information
    String templateName = alignDecl.getArg(2).getString();
    XMPtemplate templateObj = getXMPtemplate(templateName, localObjectTable);

    if (!(templateObj.isDistributed()))
      throw new XMPexception("template '" + templateName + "' is not distributed");

    int templateDim = templateObj.getDim();

    // declare array address pointer
    Ident arrayAddrId = XMPlocalDecl.addObjectId(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType), pb);

    // declare array descriptor
    Ident arrayDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + arrayName, pb);

    int arrayDim = arrayType.getNumDimensions();
    if ((arrayDim > (XMP.MAX_DIM)) || (arrayDim < 1))
      throw new XMPexception("array dimension should be less than " + (XMP.MAX_DIM + 1));

    XobjList initArrayDescFuncArgs = Xcons.List(arrayDescId.getAddr(),
                                                templateObj.getDescId().Ref(),
                                                Xcons.IntConstant(arrayDim),
                                                Xcons.SizeOf(arrayElmtType));

    Vector<Long> arraySizeVector = new Vector<Long>(arrayDim);
    Vector<Ident> gtolAccIdVector = new Vector<Ident>(arrayDim);
    for (int i = 0; i < arrayDim; i++, arrayType = arrayType.getRef()) {
      long dimSize = arrayType.getArraySize();
      if(dimSize == 0)
        throw new XMPexception("array size cannot be omitted");

      arraySizeVector.add(new Long(dimSize));
      initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));

      Ident gtolAccId = XMPlocalDecl.addObjectId(XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i,
                                                 Xtype.unsignedlonglongType, pb);
      gtolAccIdVector.add(gtolAccId);
    }

    // create/destroy local descriptor
    XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_array_desc", initArrayDescFuncArgs, pb, _globalDecl);
    XMPlocalDecl.insertDestructorCall("_XCALABLEMP_finalize_array_desc", Xcons.List(arrayDescId.Ref()), pb, _globalDecl);

    XMPalignedArray alignedArray = new XMPalignedArray(arrayName, arrayElmtType, arrayDim,
                                                       arraySizeVector, gtolAccIdVector, arrayDescId, arrayAddrId,
                                                       templateObj);
    localObjectTable.putAlignedArray(alignedArray);

    // check <align-source> list, <align-subscrip> list
    XobjList alignSourceList = (XobjList)alignDecl.getArg(1);
    XobjList alignSubscriptList = (XobjList)alignDecl.getArg(3);
    XobjList alignSubscriptVarList = (XobjList)alignSubscriptList.left();
    XobjList alignSubscriptExprList = (XobjList)alignSubscriptList.right();

    // check <align-source> list
    if (XMPutil.countElmts(alignSourceList) != arrayDim)
      throw new XMPexception("the number of <align-source>s is not the same with array dimension");

    // check <align-subscript> list
    if (XMPutil.countElmts(alignSubscriptVarList) != templateDim)
      throw new XMPexception("the number of <align-subscript>s is not the same with template dimension");

    // check ':' source/subscript
    if (XMPutil.countElmts(alignSourceList, XMP.COLON) !=
        XMPutil.countElmts(alignSubscriptVarList, XMP.COLON))
      throw new XMPexception("the number of ':' in <align-source> list is not the same with <align-subscript> list");

    // create align function calls
    int alignSourceIndex = 0;
    for (XobjArgs i = alignSourceList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSource = i.getArg().getString();

      if (alignSource.equals(XMP.ASTERISK)) {
        // do nothing
      }
      else if (alignSource.equals(XMP.COLON)) {
        if (!XMPutil.hasElmt(alignSubscriptVarList, XMP.COLON))
          throw new XMPexception("cannot find ':' in <align-subscript> list");

        int alignSubscriptIndex = XMPutil.getLastIndex(alignSubscriptVarList, XMP.COLON);
        alignSubscriptVarList.setArg(alignSubscriptIndex, null);

        declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex, null, pb);
      }
      else {
        if (XMPutil.countElmts(alignSourceList, alignSource) != 1)
          throw new XMPexception("multiple '" + alignSource + "' indicated in <align-source> list");

        if (XMPutil.hasElmt(alignSubscriptVarList, alignSource)) {
          if (XMPutil.countElmts(alignSubscriptVarList, alignSource) != 1)
            throw new XMPexception("multiple '" + alignSource + "' indicated in <align-subscript> list");

          int alignSubscriptIndex = XMPutil.getFirstIndex(alignSubscriptVarList, alignSource);
          declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
                        alignSubscriptExprList.getArg(alignSubscriptIndex), pb);
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

    XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_array_comm", initArrayCommFuncArgs, pb, _globalDecl);

    // init array address
    XobjList initArrayAddrFuncArgs = Xcons.List(arrayAddrId.getAddr(),
                                                arrayId.getAddr(),
                                                arrayDescId.Ref());
    for (int i = arrayDim - 1; i >= 0; i--)
      initArrayAddrFuncArgs.add(Xcons.Cast(Xtype.unsignedlonglongType,
                                           alignedArray.getGtolAccIdAt(i).getAddr()));

    XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_array_addr", initArrayAddrFuncArgs, pb, _globalDecl);
  }

  private void declAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                             XMPtemplate templateObj, int alignSubscriptIndex,
                             Xobject alignSubscriptExpr, PragmaBlock pb) throws XMPexception {
    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex),
                                        Xcons.IntConstant(alignSubscriptIndex));

    if (alignSubscriptExpr == null) alignFuncArgs.add(Xcons.IntConstant(0));
    else alignFuncArgs.add(alignSubscriptExpr);

    int distManner = templateObj.getDistMannerAt(alignSubscriptIndex);
    alignedArray.setDistMannerAt(distManner, alignSourceIndex);

    alignedArray.setAlignSubscriptIndexAt(alignSubscriptIndex, alignSourceIndex);
    alignedArray.setAlignSubscriptExprAt(alignSubscriptExpr, alignSourceIndex);

    switch (distManner) {
      case XMPtemplate.DUPLICATION: // FIXME how implement???
        break;
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
        {
          Ident gtolTemp0Id = XMPlocalDecl.addObjectId(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
                                                       Xtype.intType, pb);
          alignedArray.setGtolTemp0IdAt(gtolTemp0Id, alignSourceIndex);
          alignFuncArgs.add(gtolTemp0Id.getAddr());

          break;
        }
      default:
        throw new XMPexception("unknown distribute manner");
    }

    XMPlocalDecl.addConstructorCall("_XCALABLEMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
                                    alignFuncArgs, pb, _globalDecl);
  }

  // FIXME incomplete, not checked
  private void translateShadow(PragmaBlock pb) throws XMPexception {
    // check position
    checkDeclPragmaLocation(pb);

    // start translation
    XobjList shadowDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // find aligned array
    String arrayName = shadowDecl.getArg(0).getString();
    XMPalignedArray alignedArray = localObjectTable.getAlignedArray(arrayName);
    if (alignedArray == null)
      throw new XMPexception("the aligned array '" + arrayName + "' is not found in local scope");

    if (alignedArray.hasShadow())
      throw new XMPexception("the aligned array '" + arrayName + "' has the shadow declaration already");

    // init shadow
    XobjList shadowFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
    int arrayIndex = 0;
    int arrayDim = alignedArray.getDim();
    for (XobjArgs i = shadowDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      if (arrayIndex == arrayDim)
        throw new XMPexception("wrong shadow dimension indicated, too many");

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
            if (alignedArray.getDistMannerAt(arrayIndex) == XMPtemplate.DUPLICATION)
              throw new XMPexception("indicated dimension is not distributed");

            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_NORMAL)));
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, shadowBody.left()));
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, shadowBody.right()));

            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_NORMAL, shadowBody.left(), shadowBody.right()), arrayIndex);
            break;
          }
        case XMPshadow.SHADOW_FULL:
          {
            if (alignedArray.getDistMannerAt(arrayIndex) == XMPtemplate.DUPLICATION)
              throw new XMPexception("indicated dimension is not distributed");

            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_FULL)));

            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_FULL, null, null), arrayIndex);
            break;
          }
        default:
          throw new XMPexception("unknown shadow type");
      }

      arrayIndex++;
    }

    if (arrayIndex != arrayDim)
      throw new XMPexception("the number of <nodes/template-subscript> should be the same with the dimension");

    XMPlocalDecl.addConstructorCall("_XCALABLEMP_init_shadow", shadowFuncArgs, pb, _globalDecl);

    // set shadow flag
    alignedArray.setHasShadow();
  }

  private void translateTask(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList taskDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);
    BlockList taskBody = pb.getBody();

    // create function arguments
    XobjList onRef = (XobjList)taskDecl.getArg(0);
    XMPtriplet<String, Boolean, XobjList> execOnRefArgs = createExecOnRefArgs(onRef, localObjectTable);
    String execFuncSurfix = execOnRefArgs.getFirst();
    boolean splitComm = execOnRefArgs.getSecond().booleanValue();
    XobjList execFuncArgs = execOnRefArgs.getThird();

    // setup task finalizer
    Ident finFuncId = null;
    if (splitComm) finFuncId = _globalDecl.declExternFunc("_XCALABLEMP_pop_n_free_nodes");
    else           finFuncId = _globalDecl.declExternFunc("_XCALABLEMP_pop_nodes");
    setupFinalizer(taskBody, finFuncId, null);

    // create function call
    Ident execFuncId = _env.declExternIdent("_XCALABLEMP_exec_task_" + execFuncSurfix, Xtype.Function(Xtype.boolType));
    pb.replace(Bcons.IF(BasicBlock.Cond(execFuncId.Call(execFuncArgs)), taskBody, null));
  }

  private void translateTasks(PragmaBlock pb) {
    System.out.println("TASKS:" + pb.toXobject().toString());
  }

  private void translateLoop(PragmaBlock pb) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();
    BlockList loopBody = pb.getBody();

    // get block to schedule
    CforBlock schedBaseBlock = getOutermostLoopBlock(loopBody);

    // schedule loop
    if (loopDecl.getArg(0) == null) translateFollowingLoop(pb, schedBaseBlock);
    else                            translateMultipleLoop(pb, schedBaseBlock);

    // translate reduction clause
    XobjList reductionRefList = (XobjList)loopDecl.getArg(2);
    if (reductionRefList != null) {
      XobjList schedVarList = null;
      if (loopDecl.getArg(0) == null) schedVarList = Xcons.List(Xcons.String(schedBaseBlock.getInductionVar().getSym()));
      else                            schedVarList = (XobjList)loopDecl.getArg(0).copy();

      BlockList reductionBody = createReductionClauseBody(pb, reductionRefList, schedBaseBlock);
      schedBaseBlock.add(createReductionClauseBlock(pb, reductionBody, schedVarList));
    }

    // replace pragma
    pb.replace(Bcons.COMPOUND(loopBody));
  }

  private Block createReductionClauseBlock(PragmaBlock pb, BlockList reductionBody, XobjList schedVarList) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();

    XobjList onRef = (XobjList)loopDecl.getArg(1);
    String onRefObjName = onRef.getArg(0).getString();
    XobjList subscriptList = (XobjList)onRef.getArg(1);

    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);
    XMPobject onRefObj = getXMPobject(onRefObjName, localObjectTable);
    String initFuncSurfix = null;
    switch (onRefObj.getKind()) {
      case XMPobject.TEMPLATE:
        initFuncSurfix = "TEMPLATE";
        break;
      case XMPobject.NODES:
        initFuncSurfix = "NODES";
        break;
      default:
        throw new XMPexception("unknown object type");
    }

    // create function arguments
    XobjList initFuncArgs = Xcons.List(onRefObj.getDescId().Ref());

    boolean initComm = false;
    for (XobjArgs i = subscriptList.getArgs(); i != null; i = i.nextArgs()) {
      String subscript = i.getArg().getString();
      if (XMPutil.hasElmt(schedVarList, subscript))
        initFuncArgs.add(Xcons.IntConstant(0));
      else {
        initComm = true;
        initFuncArgs.add(Xcons.IntConstant(1));
      }
    }

    if (initComm) {
      // setup finalizer
      Ident finFuncId = _globalDecl.declExternFunc("_XCALABLEMP_pop_n_free_nodes");
      setupFinalizer(reductionBody, finFuncId, null);

      // create function call
      Ident initFuncId = _globalDecl.declExternFunc("_XCALABLEMP_init_reduce_comm_" + initFuncSurfix);
      reductionBody.insert(initFuncId.Call(initFuncArgs));
    }

    return Bcons.COMPOUND(reductionBody);
  }

  private void translateFollowingLoop(PragmaBlock pb, CforBlock schedBaseBlock) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();

    // schedule loop
    scheduleLoop(pb, schedBaseBlock, schedBaseBlock);
  }

  private void translateMultipleLoop(PragmaBlock pb, CforBlock schedBaseBlock) throws XMPexception {
    // start translation
    XobjList loopDecl = (XobjList)pb.getClauses();
    BlockList loopBody = pb.getBody();

    // iterate index variable list
    XobjList loopVarList = (XobjList)loopDecl.getArg(0);
    Vector<CforBlock> loopVector = new Vector<CforBlock>(XMPutil.countElmts(loopVarList));
    for (XobjArgs i = loopVarList.getArgs(); i != null; i = i.nextArgs())
      loopVector.add(findLoopBlock(loopBody, i.getArg().getString()));

    // schedule loop
    Iterator<CforBlock> it = loopVector.iterator();
    while (it.hasNext()) {
      CforBlock forBlock = it.next();
      scheduleLoop(pb, forBlock, schedBaseBlock);
    }
  }

  private BlockList createReductionClauseBody(PragmaBlock pb, XobjList reductionRefList,
                                              CforBlock schedBaseBlock) throws XMPexception {
    // create init block
    Ident getRankFuncId = _globalDecl.declExternFunc("_XCALABLEMP_get_execution_nodes_rank", Xtype.intType);
    IfBlock reductionInitIfBlock = (IfBlock)Bcons.IF(BasicBlock.Cond(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, getRankFuncId.Call(null),
                                                                                                       Xcons.IntConstant(0))),
                                                     Bcons.emptyBody(), Bcons.emptyBody());

    // create function call
    Iterator<Xobject> it = reductionRefList.iterator();
    BlockList reductionBody = Bcons.emptyBody();
    while (it.hasNext()) {
      XobjList reductionRef = (XobjList)it.next();
      Vector<XobjList> reductionFuncArgsList = createReductionArgsList(reductionRef, pb,
                                                                       true, schedBaseBlock, reductionInitIfBlock);
      String reductionFuncType = createReductionFuncType(reductionRef, pb);

      reductionBody.add(createReductionFuncCallBlock(false, reductionFuncType + "_CLAUSE",
                                                     null, reductionFuncArgsList));
    }

    schedBaseBlock.insert(reductionInitIfBlock);

    return reductionBody;
  }

  private void scheduleLoop(PragmaBlock pb, CforBlock forBlock, CforBlock schedBaseBlock) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(schedBaseBlock);

    // analyze <on-ref>
    Xobject onRef = loopDecl.getArg(1);
    String onRefObjName = onRef.getArg(0).getString();
    XMPobject onRefObj = getXMPobject(onRefObjName, localObjectTable);
    switch (onRefObj.getKind()) {
      case XMPobject.TEMPLATE:
        {
          XMPtemplate onRefTemplate = (XMPtemplate)onRefObj;
          if (!(onRefTemplate.isDistributed()))
            throw new XMPexception("template '" + onRefObjName + "' is not distributed");

          callLoopSchedFuncTemplate(onRefTemplate, (XobjList)onRef.getArg(1), forBlock, schedBaseBlock);
        } break;
      case XMPobject.NODES:
        callLoopSchedFuncNodes((XMPnodes)onRefObj, (XobjList)onRef.getArg(1), forBlock, schedBaseBlock);
        break;
      default:
        throw new XMPexception("unknown object type");
    }
  }

  private static CforBlock getOutermostLoopBlock(BlockList body) throws XMPexception {
    Block b = body.getHead();
    if (b != null) {
      if (b.Opcode() == Xcode.FOR_STATEMENT) {
        LineNo blockLnObj = b.getLineNo();

        // XXX too strict?
        if (b.getNext() != null)
          throw new XMPexception("only one loop statement is allowed in loop directive");

        CforBlock forBlock = (CforBlock)b;
        forBlock.Canonicalize();
        if (!(forBlock.isCanonical()))
          throw new XMPexception("loop statement is not canonical");

        return forBlock;
      }
      else if (b.Opcode() == Xcode.COMPOUND_STATEMENT)
        return getOutermostLoopBlock(b.getBody());
    }

    throw new XMPexception("cannot find the loop statement");
  }

  private static CforBlock findLoopBlock(BlockList body, String loopVarName) throws XMPexception {
    Block b = body.getHead();
    if (b != null) {
      switch (b.Opcode()) {
        case FOR_STATEMENT:
          {
            CforBlock forBlock = (CforBlock)b;
            forBlock.Canonicalize();
            if (!(forBlock.isCanonical()))
              throw new XMPexception("loop is not canonical");

            if (forBlock.getInductionVar().getSym().equals(loopVarName))
              return (CforBlock)b;
            else
              return findLoopBlock(forBlock.getBody(), loopVarName);
          }
        case COMPOUND_STATEMENT:
          return findLoopBlock(b.getBody(), loopVarName);
        case XMP_PRAGMA:
        case OMP_PRAGMA:
          throw new XMPexception("reached to a openmp/xcalablemp directive");
      }
    }

    throw new XMPexception("cannot find the loop statement");
  }

  private void callLoopSchedFuncTemplate(XMPtemplate templateObj, XobjList templateSubscriptList,
                                         CforBlock forBlock, CforBlock schedBaseBlock) throws XMPexception {
    Xobject loopIndex = forBlock.getInductionVar();
    Xtype loopIndexType = loopIndex.Type();

    if (!XMPutil.isIntegerType(loopIndexType))
      throw new XMPexception("loop index variable has a non-integer type");

    String funcTypeSurfix = XMPutil.getTypeName(loopIndexType);
    String loopIndexName = loopIndex.getSym();

    XobjList funcArgs = Xcons.List();
    funcArgs.add(forBlock.getLowerBound());
    funcArgs.add(forBlock.getUpperBound());
    funcArgs.add(forBlock.getStep());

    int templateIndex = 0;
    int templateDim = templateObj.getDim();
    XobjInt templateIndexArg = null;
    int distManner = 0;
    String distMannerString = null;
    for (XobjArgs i = templateSubscriptList.getArgs(); i != null; i = i.nextArgs()) {
      if (templateIndex >= templateDim)
        throw new XMPexception("wrong template dimensions, too many");

      String s = i.getArg().getString();
      if (s.equals(loopIndexName)) {
        if (templateIndexArg != null)
          throw new XMPexception("loop index '" + loopIndexName + "' is already described");

        templateIndexArg = Xcons.IntConstant(templateIndex);
        distManner = templateObj.getDistMannerAt(templateIndex);
        if (distManner == XMPtemplate.DUPLICATION) // FIXME how implement???
          throw new XMPexception("indicated template dimension is not distributed");
        else distMannerString = templateObj.getDistMannerStringAt(templateIndex);
      }

      templateIndex++;
    }

    if(templateIndexArg == null)
      throw new XMPexception("cannot find index '" + loopIndexName + "' reference in <on-ref>");

    if(templateIndex != templateDim)
      throw new XMPexception("wrong template dimensions, too few");

    Ident parallelInitId = declIdentWithBlock(schedBaseBlock,
                                              "_XCALABLEMP_loop_init_" + loopIndexName, loopIndexType);
    Ident parallelCondId = declIdentWithBlock(schedBaseBlock,
                                              "_XCALABLEMP_loop_cond_" + loopIndexName, loopIndexType);
    Ident parallelStepId = declIdentWithBlock(schedBaseBlock,
                                              "_XCALABLEMP_loop_step_" + loopIndexName, loopIndexType);

    forBlock.setLowerBound(parallelInitId.Ref());
    forBlock.setUpperBound(parallelCondId.Ref());
    forBlock.setStep(parallelStepId.Ref());

    forBlock.getCondBBlock().setExpr(Xcons.binaryOp(Xcode.LOG_LT_EXPR, loopIndex, parallelCondId.Ref()));

    funcArgs.add(parallelInitId.getAddr());
    funcArgs.add(parallelCondId.getAddr());
    funcArgs.add(parallelStepId.getAddr());

    funcArgs.add(templateObj.getDescId().Ref());
    funcArgs.add(templateIndexArg);

    Ident funcId = _globalDecl.declExternFunc("_XCALABLEMP_sched_loop_template_" + distMannerString + "_" + funcTypeSurfix);

    schedBaseBlock.insert(funcId.Call(funcArgs));
  }

  private void callLoopSchedFuncNodes(XMPnodes nodesObj, XobjList nodesSubscriptList,
                                      CforBlock forBlock, CforBlock schedBaseBlock) throws XMPexception {
    Xobject loopIndex = forBlock.getInductionVar();
    Xtype loopIndexType = loopIndex.Type();

    if (!XMPutil.isIntegerType(loopIndexType))
      throw new XMPexception("loop index variable has a non-integer type");

    String funcTypeSurfix = XMPutil.getTypeName(loopIndexType);
    String loopIndexName = loopIndex.getSym();

    XobjList funcArgs = Xcons.List();
    funcArgs.add(forBlock.getLowerBound());
    funcArgs.add(forBlock.getUpperBound());
    funcArgs.add(forBlock.getStep());

    int nodesIndex = 0;
    int nodesDim = nodesObj.getDim();
    XobjInt nodesIndexArg = null;
    for (XobjArgs i = nodesSubscriptList.getArgs(); i != null; i = i.nextArgs()) {
      if (nodesIndex >= nodesDim)
        throw new XMPexception("wrong nodes dimensions, too many");

      String s = i.getArg().getString();
      if (s.equals(loopIndexName)) {
        if (nodesIndexArg != null)
          throw new XMPexception("loop index '" + loopIndexName + "' is already described");

        nodesIndexArg = Xcons.IntConstant(nodesIndex);
      }

      nodesIndex++;
    }

    if(nodesIndexArg == null)
      throw new XMPexception("cannot find index '" + loopIndexName + "' reference in <on-ref>");

    if(nodesIndex != nodesDim)
      throw new XMPexception("wrong nodes dimensions, too few");

    Ident parallelInitId = declIdentWithBlock(schedBaseBlock,
                                              "_XCALABLEMP_loop_init_" + loopIndexName, loopIndexType);
    Ident parallelCondId = declIdentWithBlock(schedBaseBlock,
                                              "_XCALABLEMP_loop_cond_" + loopIndexName, loopIndexType);
    Ident parallelStepId = declIdentWithBlock(schedBaseBlock,
                                              "_XCALABLEMP_loop_step_" + loopIndexName, loopIndexType);

    forBlock.setLowerBound(parallelInitId.Ref());
    forBlock.setUpperBound(parallelCondId.Ref());
    forBlock.setStep(parallelStepId.Ref());

    forBlock.getCondBBlock().setExpr(Xcons.binaryOp(Xcode.LOG_LT_EXPR, loopIndex, parallelCondId.Ref()));

    funcArgs.add(parallelInitId.getAddr());
    funcArgs.add(parallelCondId.getAddr());
    funcArgs.add(parallelStepId.getAddr());

    funcArgs.add(nodesObj.getDescId().Ref());
    funcArgs.add(nodesIndexArg);

    Ident funcId = _globalDecl.declExternFunc("_XCALABLEMP_sched_loop_nodes_" + funcTypeSurfix);

    schedBaseBlock.insert(funcId.Call(funcArgs));
  }

  private Ident declIdentWithBlock(Block b, String identName, Xtype type) {
    BlockList bl = b.getParent();

    // FIXME consider variable scope
    return bl.declLocalIdent(identName, type);
  }

  private void translateReflect(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList reflectDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);
    BlockList reflectFuncBody = Bcons.emptyBody();

    XobjList arrayList = (XobjList)reflectDecl.getArg(0);
    for (XobjArgs iter = arrayList.getArgs(); iter != null; iter = iter.nextArgs()) {
      String arrayName = iter.getArg().getString();
      XMPalignedArray alignedArray = getXMPalignedArray(arrayName, localObjectTable);

      if (!alignedArray.hasShadow()) {
        throw new XMPexception("the aligned array '" + arrayName + "' has no shadow declaration");
      }

      int arrayDim = alignedArray.getDim();
      for (int i = 0; i < arrayDim; i++) {
        XMPshadow shadowObj = alignedArray.getShadowAt(i);
        switch (shadowObj.getType()) {
          case XMPshadow.SHADOW_NONE:
            break;
          case XMPshadow.SHADOW_NORMAL:
            createReflectNormalShadowFunc(pb, alignedArray, i, reflectFuncBody);
            break;
          case XMPshadow.SHADOW_FULL:
            // FIXME not implemented yet
            throw new XMPexception("not implemented yet");
          default:
            throw new XMPexception("unknown shadow type");
        }
      }
    }

    pb.replace(Bcons.COMPOUND(reflectFuncBody));
  }

  // FIXME implementing now
  private void createReflectNormalShadowFunc(PragmaBlock pb,
                                             XMPalignedArray alignedArray, int arrayIndex,
                                             BlockList reflectFuncBody) {
    String arrayName = alignedArray.getName();
    Xtype arrayType = alignedArray.getType();
    Xtype arrayPtype = Xtype.Pointer(arrayType);
    String arrayTypeName = XMPutil.getTypeName(arrayType);

    // decl buffers
    Ident loSendId = reflectFuncBody.declLocalIdent("_XCALABLEMP_reflect_LO_SEND_" + arrayName, arrayPtype);
    Ident loRecvId = reflectFuncBody.declLocalIdent("_XCALABLEMP_reflect_LO_RECV_" + arrayName, arrayPtype);
    Ident hiSendId = reflectFuncBody.declLocalIdent("_XCALABLEMP_reflect_HI_SEND_" + arrayName, arrayPtype);
    Ident hiRecvId = reflectFuncBody.declLocalIdent("_XCALABLEMP_reflect_HI_RECV_" + arrayName, arrayPtype);

    // pack shadow
    Ident packFuncId = null;
    XobjList packFuncArgs = Xcons.List(loSendId.getAddr(), hiSendId.getAddr(), alignedArray.getAddrId().Ref(),
                                       alignedArray.getDescId().Ref(), Xcons.IntConstant(arrayIndex));
    if (arrayType.getKind() == Xtype.BASIC) {
      packFuncId = _globalDecl.declExternFunc("_XCALABLEMP_pack_shadow_NORMAL_" + "BASIC");
      packFuncArgs.add(Xcons.IntConstant(arrayType.getBasicType() + 200));
    }
    else {
      packFuncId = _globalDecl.declExternFunc("_XCALABLEMP_pack_shadow_NORMAL_" + "GENERAL");
    }

    reflectFuncBody.add(Bcons.Statement(packFuncId.Call(packFuncArgs)));

    // exchange shadow
    Ident exchangeFuncId = _globalDecl.declExternFunc("_XCALABLEMP_exchange_shadow_NORMAL");
    XobjList exchangeFuncArgs = Xcons.List(loRecvId.getAddr(), hiRecvId.getAddr(), loSendId.Ref(), hiSendId.Ref());
    exchangeFuncArgs.add(alignedArray.getDescId().Ref());
    exchangeFuncArgs.add(Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(exchangeFuncId.Call(exchangeFuncArgs)));

    // unpack shadow
    Ident unpackFuncId = null;
    XobjList unpackFuncArgs = Xcons.List(loRecvId.Ref(), hiRecvId.Ref(), alignedArray.getAddrId().Ref(),
                                         alignedArray.getDescId().Ref(), Xcons.IntConstant(arrayIndex));
    if (arrayType.getKind() == Xtype.BASIC) {
      unpackFuncId = _globalDecl.declExternFunc("_XCALABLEMP_unpack_shadow_NORMAL_" + "BASIC");
      unpackFuncArgs.add(Xcons.IntConstant(arrayType.getBasicType() + 200));
    }
    else {
      unpackFuncId = _globalDecl.declExternFunc("_XCALABLEMP_unpack_shadow_NORMAL_" + "GENERAL");
    }

    reflectFuncBody.add(Bcons.Statement(unpackFuncId.Call(unpackFuncArgs)));
  }

  private void translateBarrier(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList barrierDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // create function call
    XobjList onRef = (XobjList)barrierDecl.getArg(0);
    if (onRef == null) pb.replace(createFuncCallBlock("_XCALABLEMP_barrier_EXEC", null));
    else {
      XMPtriplet<String, Boolean, XobjList> execOnRefArgs = createExecOnRefArgs(onRef, localObjectTable);
      String execFuncSurfix = execOnRefArgs.getFirst();
      boolean splitComm = execOnRefArgs.getSecond().booleanValue();
      XobjList execFuncArgs = execOnRefArgs.getThird();
      if (splitComm) {
        BlockList barrierBody = Bcons.blockList(createFuncCallBlock("_XCALABLEMP_barrier_EXEC", null));

        // setup barrier finalizer
        setupFinalizer(barrierBody, _globalDecl.declExternFunc("_XCALABLEMP_pop_n_free_nodes"), null);

        // create function call
        Ident execFuncId = _env.declExternIdent("_XCALABLEMP_exec_task_" + execFuncSurfix, Xtype.Function(Xtype.boolType));
        pb.replace(Bcons.IF(BasicBlock.Cond(execFuncId.Call(execFuncArgs)), barrierBody, null));
      }
      else pb.replace(createFuncCallBlock("_XCALABLEMP_barrier_" + execFuncSurfix, execFuncArgs));
    }
  }

  private Block createFuncCallBlock(String funcName, XobjList funcArgs) {
    Ident funcId = _globalDecl.declExternFunc(funcName);
    return Bcons.Statement(funcId.Call(funcArgs));
  }

  private void translateReduction(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList reductionDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // create function arguments
    XobjList reductionRef = (XobjList)reductionDecl.getArg(0);
    Vector<XobjList> reductionFuncArgsList = createReductionArgsList(reductionRef, pb,
                                                                     false, null, null);
    String reductionFuncType = createReductionFuncType(reductionRef, pb);

    // create function call
    XobjList onRef = (XobjList)reductionDecl.getArg(1);
    if (onRef == null) {
      pb.replace(createReductionFuncCallBlock(true, reductionFuncType + "_EXEC", null, reductionFuncArgsList));
    }
    else {
      XMPtriplet<String, Boolean, XobjList> execOnRefArgs = createExecOnRefArgs(onRef, localObjectTable);
      String execFuncSurfix = execOnRefArgs.getFirst();
      boolean splitComm = execOnRefArgs.getSecond().booleanValue();
      XobjList execFuncArgs = execOnRefArgs.getThird();
      if (splitComm) {
        BlockList reductionBody = Bcons.blockList(createReductionFuncCallBlock(true, reductionFuncType + "_EXEC",
                                                                               null, reductionFuncArgsList));

        // setup reduction finalizer
        setupFinalizer(reductionBody, _globalDecl.declExternFunc("_XCALABLEMP_pop_n_free_nodes"), null);

        // create function call
        Ident execFuncId = _env.declExternIdent("_XCALABLEMP_exec_task_" + execFuncSurfix, Xtype.Function(Xtype.boolType));
        pb.replace(Bcons.IF(BasicBlock.Cond(execFuncId.Call(execFuncArgs)), reductionBody, null));
      }
      else {
        pb.replace(createReductionFuncCallBlock(false, reductionFuncType + "_" + execFuncSurfix,
                                                execFuncArgs.operand(), reductionFuncArgsList));
      }
    }
  }

  private String createReductionFuncType(XobjList reductionRef, PragmaBlock pb) throws XMPexception {
    XobjInt reductionOp = (XobjInt)reductionRef.getArg(0);
    switch (reductionOp.getInt()) {
      case XMPcollective.REDUCE_SUM:
      case XMPcollective.REDUCE_PROD:
      case XMPcollective.REDUCE_BAND:
      case XMPcollective.REDUCE_LAND:
      case XMPcollective.REDUCE_BOR:
      case XMPcollective.REDUCE_LOR:
      case XMPcollective.REDUCE_BXOR:
      case XMPcollective.REDUCE_LXOR:
      case XMPcollective.REDUCE_MAX:
      case XMPcollective.REDUCE_MIN:
        return new String("reduce");
      case XMPcollective.REDUCE_FIRSTMAX:
      case XMPcollective.REDUCE_FIRSTMIN:
      case XMPcollective.REDUCE_LASTMAX:
      case XMPcollective.REDUCE_LASTMIN:
        return new String("reduce_FLMM");
      default:
        throw new XMPexception("unknown reduce operation");
    }
  }

  private Vector<XobjList> createReductionArgsList(XobjList reductionRef, PragmaBlock pb,
                                                   boolean isClause,
                                                   CforBlock schedBaseBlock, IfBlock reductionInitIfBlock) throws XMPexception {
    Vector<XobjList> returnVector = new Vector<XobjList>();

    XobjInt reductionOp = (XobjInt)reductionRef.getArg(0);

    XobjList reductionSpecList = (XobjList)reductionRef.getArg(1);
    for (XobjArgs i = reductionSpecList.getArgs(); i != null; i = i.nextArgs()) {
      XobjList reductionSpec = (XobjList)i.getArg();
      String specName = reductionSpec.getArg(0).getString();

      XMPpair<Ident, Xtype> typedSpec = findTypedVar(specName, pb);
      Ident specId = typedSpec.getFirst();
      Xtype specType = typedSpec.getSecond();

      boolean isArray = false;
      XobjLong count = null;
      XobjInt elmtType = null;
      BasicType basicSpecType = null;
      switch (specType.getKind()) {
        case Xtype.BASIC:
          {
            basicSpecType = (BasicType)specType;
            checkReductionType(specName, basicSpecType);

            count = Xcons.LongLongConstant(0, 1);
            elmtType = Xcons.IntConstant(basicSpecType.getBasicType() + 200);
          } break;
        case Xtype.ARRAY:
          {
            isArray = true;
            ArrayType arraySpecType = (ArrayType)specType;
            if (arraySpecType.getArrayElementType().getKind() != Xtype.BASIC)
              throw new XMPexception("array '" + specName + "' has has a wrong data type for reduction");

            basicSpecType = (BasicType)arraySpecType.getArrayElementType();
            checkReductionType(specName, basicSpecType);

            count = Xcons.LongLongConstant(0, getArrayElmtCount(arraySpecType));
            elmtType = Xcons.IntConstant(basicSpecType.getBasicType() + 200);
          } break;
        default:
          throw new XMPexception("'" + specName + "' has a wrong data type for reduction");
      }

      Xobject specRef = null;
      if (isArray) specRef = specId.Ref();
      else         specRef = specId.getAddr();

      XobjList reductionFuncArgs = Xcons.List(specRef, count, elmtType, reductionOp);

      // declare temp variable for reduction
      if (isClause) {
        String tempName = new String("_XCALABLEMP_reduce_temp_" + specName);
        Ident tempId = declReductionTempIdent(pb, specName, tempName, specType);
        if (isArray) {
          reductionFuncArgs.cons(tempId.Ref());
          createReductionInitStatement(tempId, specId, true, count, basicSpecType, reductionOp.getInt(),
                                       schedBaseBlock, reductionInitIfBlock);
        }
        else {
          reductionFuncArgs.cons(tempId.getAddr());
          createReductionInitStatement(tempId, specId, false, null, basicSpecType, reductionOp.getInt(),
                                       schedBaseBlock, reductionInitIfBlock);
        }

        // rewrite reduction variable
        BasicBlockExprIterator iter = new BasicBlockExprIterator(schedBaseBlock);
        for (iter.init(); !iter.end(); iter.next())
          rewriteSymbolName(iter.getExpr(), specName, tempName);
      }

      // add extra args for (firstmax, firstmin, lastmax, lastmin) if needed
      createFLMMreductionArgs(reductionOp.getInt(), count.getLongLow(),
                              (XobjList)reductionSpec.getArg(1), reductionFuncArgs, pb);

      returnVector.add(reductionFuncArgs);
    }

    return returnVector;
  }

  private void createReductionInitStatement(Ident tempId, Ident varId,
                                            boolean isArray, XobjLong count, BasicType type, int reductionOp,
                                            CforBlock schedBaseBlock, IfBlock reductionIfBlock) throws XMPexception {
    BlockList masterPart = reductionIfBlock.getThenBody();
    BlockList otherPart = reductionIfBlock.getElseBody();

    Xobject statement = null;
    if (isArray) {
      // master part
      Ident masterLoopIndexId = declIdentWithBlock(schedBaseBlock, XMPuniqueName.getTempName(), Xtype.unsignedlonglongType);
      Xobject masterInitValueObj = Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.Cast(Xtype.Pointer(type), varId.Ref()),
                                                                   masterLoopIndexId.Ref()));
      masterPart.add(createReductionArrayInit(tempId, masterInitValueObj,
                                              count, type, masterLoopIndexId));

      // other part
      Ident otherLoopIndexId = declIdentWithBlock(schedBaseBlock, XMPuniqueName.getTempName(), Xtype.unsignedlonglongType);
      otherPart.add(createReductionArrayInit(tempId, createReductionInitValueObj(varId, type, reductionOp),
                                             count, type, otherLoopIndexId));
    }
    else {
      // master part
      masterPart.add(Xcons.Set(tempId.Ref(), varId.Ref()));

      // other part
      otherPart.add(Xcons.Set(tempId.Ref(), createReductionInitValueObj(varId, type, reductionOp)));
    }
  }

  private Block createReductionArrayInit(Ident tempId, Xobject initValueObj, XobjLong count, BasicType type,
                                         Ident loopIndexId) {
    Xobject loopIndexRef = loopIndexId.Ref();

    Xobject tempArrayRef = Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.Cast(Xtype.Pointer(type), tempId.Ref()),
                                                                            loopIndexRef));

    Block loopBlock = Bcons.FOR(Xcons.Set(loopIndexRef, Xcons.IntConstant(0)),
                                Xcons.binaryOp(Xcode.LOG_LT_EXPR, loopIndexRef, count),
                                Xcons.asgOp(Xcode.ASG_PLUS_EXPR, loopIndexRef, Xcons.IntConstant(1)),
                                Bcons.Statement(Xcons.Set(tempArrayRef, initValueObj)));

    return loopBlock;
  }

  // FIXME
  private Xobject createReductionInitValueObj(Ident varId, BasicType type, int reductionOp) throws XMPexception {
    Xobject varRef = varId.Ref();
    Xobject intZero = Xcons.IntConstant(0);
    Xobject intOne = Xcons.IntConstant(1);
    Xobject floatZero = Xcons.FloatConstant(0.);
    Xobject floatOne = Xcons.FloatConstant(1.);

    switch (type.getBasicType()) {
      case BasicType.BOOL:
        // FIXME correct???
        return selectReductionInitValueObj(reductionOp, varRef, intZero, intOne);
      case BasicType.CHAR:
      case BasicType.UNSIGNED_CHAR:
      case BasicType.SHORT:
      case BasicType.UNSIGNED_SHORT:
      case BasicType.INT:
      case BasicType.UNSIGNED_INT:
      case BasicType.LONG:
      case BasicType.UNSIGNED_LONG:
      case BasicType.LONGLONG:
      case BasicType.UNSIGNED_LONGLONG:
        return selectReductionInitValueObj(reductionOp, varRef, intZero, intOne);
      case BasicType.FLOAT:
      case BasicType.DOUBLE:
      case BasicType.LONG_DOUBLE:
        return selectReductionInitValueObj(reductionOp, varRef, floatZero, floatOne);
      case BasicType.FLOAT_IMAGINARY:
      case BasicType.DOUBLE_IMAGINARY:
      case BasicType.LONG_DOUBLE_IMAGINARY:
      case BasicType.FLOAT_COMPLEX:
      case BasicType.DOUBLE_COMPLEX:
      case BasicType.LONG_DOUBLE_COMPLEX:
        // FIXME not implemented yet
        throw new XMPexception("not implemented yet");
      default:
        throw new XMPexception("wrong data type for reduction");
    }
  }

  // FIXME needs type checking
  private Xobject selectReductionInitValueObj(int reductionOp, Xobject varRef, Xobject zero, Xobject one) throws XMPexception {
    switch (reductionOp) {
      case XMPcollective.REDUCE_SUM:
        return zero;
      case XMPcollective.REDUCE_PROD:
        return one;
      case XMPcollective.REDUCE_BAND:
      case XMPcollective.REDUCE_LAND:
      case XMPcollective.REDUCE_BOR:
      case XMPcollective.REDUCE_LOR:
      case XMPcollective.REDUCE_MAX:
      case XMPcollective.REDUCE_MIN:
      case XMPcollective.REDUCE_FIRSTMAX:
      case XMPcollective.REDUCE_FIRSTMIN:
      case XMPcollective.REDUCE_LASTMAX:
      case XMPcollective.REDUCE_LASTMIN:
        return varRef;
      case XMPcollective.REDUCE_BXOR:
        return Xcons.unaryOp(Xcode.BIT_NOT_EXPR, varRef);
      case XMPcollective.REDUCE_LXOR:
        return Xcons.unaryOp(Xcode.LOG_NOT_EXPR, varRef);
      default:
        throw new XMPexception("unknown reduce operation");
    }
  }

  private Ident declReductionTempIdent(Block b, String oldName, String newName, Xtype type) {
    BlockList bl = b.getParent();
    Ident newId = null;
    Ident oldId = bl.findLocalIdent(oldName);
    if (oldId == null) {
      oldId = _env.findVarIdent(oldName);
      if (oldId != null)
        newId = _env.declStaticIdent(newName, type);
      else
        newId = bl.declLocalIdent(newName, type);
    }
    else newId = bl.declLocalIdent(newName, type);

    return newId;
  }

  private void rewriteSymbolName(Xobject expr, String oldName, String newName) {
    if (expr == null) return;

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) continue;

      if (myExpr instanceof XobjString) {
        XobjString symbol = (XobjString)myExpr;
        if (symbol.getSym().equals(oldName)) symbol.setSym(newName);
      }
    }
  }

  private void createFLMMreductionArgs(int op, long count, XobjList locationVars,
                                       XobjList funcArgs, PragmaBlock pb) throws XMPexception {
    switch (op) {
      case XMPcollective.REDUCE_SUM:
      case XMPcollective.REDUCE_PROD:
      case XMPcollective.REDUCE_BAND:
      case XMPcollective.REDUCE_LAND:
      case XMPcollective.REDUCE_BOR:
      case XMPcollective.REDUCE_LOR:
      case XMPcollective.REDUCE_BXOR:
      case XMPcollective.REDUCE_LXOR:
      case XMPcollective.REDUCE_MAX:
      case XMPcollective.REDUCE_MIN:
        return;
      case XMPcollective.REDUCE_FIRSTMAX:
      case XMPcollective.REDUCE_FIRSTMIN:
      case XMPcollective.REDUCE_LASTMAX:
      case XMPcollective.REDUCE_LASTMIN:
        break;
      default:
        throw new XMPexception("unknown reduce operation");
    }

    funcArgs.add(Xcons.IntConstant(XMPutil.countElmts(locationVars)));

    // check <location-variables> and add to funcArgs
    for (XobjArgs i = locationVars.getArgs(); i != null; i = i.nextArgs()) {
      String varName = i.getArg().getString();

      XMPpair<Ident, Xtype> typedVar = findTypedVar(varName, pb);
      Ident varId = typedVar.getFirst();
      Xtype varType = typedVar.getSecond();

      switch (varType.getKind()) {
        case Xtype.BASIC:
          {
            if (!XMPutil.isIntegerType(varType))
              throw new XMPexception("'" + varName + "' should have a integer type for reduction");

            BasicType basicVarType = (BasicType)varType;
            
            if (count != 1)
              throw new XMPexception("'" + varName + "' should be a scalar variable");

            funcArgs.add(Xcons.Cast(Xtype.voidPtrType, varId.getAddr()));
            funcArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(basicVarType.getBasicType() + 200)));
          } break;
        case Xtype.ARRAY:
          {
            ArrayType arrayVarType = (ArrayType)varType;
            if (arrayVarType.getArrayElementType().getKind() != Xtype.BASIC)
              throw new XMPexception("array '" + varName + "' has has a wrong data type for reduction");

            BasicType basicVarType = (BasicType)arrayVarType.getArrayElementType();

            if (!XMPutil.isIntegerType(basicVarType))
              throw new XMPexception("'" + varName + "' should have a integer type for reduction");

            if (count != getArrayElmtCount(arrayVarType))
              throw new XMPexception("the size of '" + varName + "' is not same with the <reduction-variable>");

            funcArgs.add(Xcons.Cast(Xtype.voidPtrType, varId.Ref()));
            funcArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(basicVarType.getBasicType() + 200)));
          } break;
        default:
          throw new XMPexception("'" + varName + "' has a wrong data type for reduction");
      }
    }
  }

  private void checkReductionType(String name, BasicType type) throws XMPexception {
    switch (type.getBasicType()) {
      case BasicType.BOOL:
      case BasicType.CHAR:
      case BasicType.UNSIGNED_CHAR:
      case BasicType.SHORT:
      case BasicType.UNSIGNED_SHORT:
      case BasicType.INT:
      case BasicType.UNSIGNED_INT:
      case BasicType.LONG:
      case BasicType.UNSIGNED_LONG:
      case BasicType.LONGLONG:
      case BasicType.UNSIGNED_LONGLONG:
      case BasicType.FLOAT:
      case BasicType.DOUBLE:
      case BasicType.LONG_DOUBLE:
      case BasicType.FLOAT_IMAGINARY:
      case BasicType.DOUBLE_IMAGINARY:
      case BasicType.LONG_DOUBLE_IMAGINARY:
      case BasicType.FLOAT_COMPLEX:
      case BasicType.DOUBLE_COMPLEX:
      case BasicType.LONG_DOUBLE_COMPLEX:
        break;
      default:
        throw new XMPexception("'" + name + "' has a wrong data type for reduction");
    }
  }

  private Block createReductionFuncCallBlock(boolean isMacroFunc, String funcType,
                                             Xobject execDesc, Vector<XobjList> funcArgsList) {
    Ident funcId = null;
    if (isMacroFunc) funcId = XMP.getMacroId("_XCALABLEMP_M_" + funcType.toUpperCase());
    else             funcId = _globalDecl.declExternFunc("_XCALABLEMP_" + funcType);

    BlockList funcCallList = Bcons.emptyBody();
    Iterator<XobjList> it = funcArgsList.iterator();
    while (it.hasNext()) {
      XobjList funcArgs = it.next();
      if (execDesc != null) funcArgs.cons(execDesc);

      funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
    }
    return Bcons.COMPOUND(funcCallList);
  }

  private XMPpair<Ident, Xtype> findTypedVar(String name, PragmaBlock pb) throws XMPexception {
    Ident id = pb.findVarIdent(name);
    if (id == null)
      throw new XMPexception("cannot find '" + name + "'");

    Xtype type = id.Type();
    if (type == null)
      throw new XMPexception("'" + name + "' has no type");

    return new XMPpair<Ident, Xtype>(id, type);
  }

  public long getArrayElmtCount(Xtype type) {
    if (type.isArray()) {
      ArrayType arrayType = (ArrayType)type;
      long arraySize = arrayType.getArraySize();
      return arraySize * getArrayElmtCount(arrayType.getRef());
    }
    else return 1;
  }

  private void translateBcast(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList bcastDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);

    // create function arguments
    XobjList varList = (XobjList)bcastDecl.getArg(0);
    Vector<XobjList> bcastArgsList = createBcastArgsList(varList, pb);

    // create function call
    XobjList fromRef = (XobjList)bcastDecl.getArg(1);
    XMPpair<String, XobjList> execFromRefArgs = null;
    if (fromRef != null) execFromRefArgs = createExecFromRefArgs(fromRef, localObjectTable);

    XobjList onRef = (XobjList)bcastDecl.getArg(2);
    if (onRef == null) pb.replace(createBcastFuncCallBlock(true, "EXEC", null, bcastArgsList, execFromRefArgs));
    else {
      XMPtriplet<String, Boolean, XobjList> execOnRefArgs = createExecOnRefArgs(onRef, localObjectTable);

      String execFuncSurfix = execOnRefArgs.getFirst();
      boolean splitComm = execOnRefArgs.getSecond().booleanValue();
      XobjList execFuncArgs = execOnRefArgs.getThird();
      if (splitComm) {
        BlockList bcastBody = Bcons.blockList(createBcastFuncCallBlock(true, "EXEC",
                                                                       null, bcastArgsList, execFromRefArgs));

        // setup reduction finalizer
        setupFinalizer(bcastBody, _globalDecl.declExternFunc("_XCALABLEMP_pop_n_free_nodes"), null);

        // create function call
        Ident execFuncId = _env.declExternIdent("_XCALABLEMP_exec_task_" + execFuncSurfix, Xtype.Function(Xtype.boolType));
        pb.replace(Bcons.IF(BasicBlock.Cond(execFuncId.Call(execFuncArgs)), bcastBody, null));
      }
      else {
        pb.replace(createBcastFuncCallBlock(false, execFuncSurfix,
                                            execFuncArgs.operand(), bcastArgsList, execFromRefArgs));
      }
    }
  }

  private Vector<XobjList> createBcastArgsList(XobjList varList, PragmaBlock pb) throws XMPexception {
    Vector<XobjList> returnVector = new Vector<XobjList>();

    for (XobjArgs i = varList.getArgs(); i != null; i = i.nextArgs()) {
      String varName = i.getArg().getString();

      XMPpair<Ident, Xtype> typedSpec = findTypedVar(varName, pb);
      Ident varId = typedSpec.getFirst();
      Xtype varType = typedSpec.getSecond();

      XobjLong count = null;
      switch (varType.getKind()) {
        case Xtype.BASIC:
        case Xtype.STRUCT:
        case Xtype.UNION:
          {
            count = Xcons.LongLongConstant(0, 1);
            returnVector.add(Xcons.List(varId.getAddr(), count, Xcons.SizeOf(varType)));
          } break;
        case Xtype.ARRAY:
          {
            ArrayType arrayVarType = (ArrayType)varType;
            switch (arrayVarType.getArrayElementType().getKind()) {
              case Xtype.BASIC:
              case Xtype.STRUCT:
              case Xtype.UNION:
                break;
              default:
                throw new XMPexception("array '" + varName + "' has has a wrong data type for broadcast");
            }

            count = Xcons.LongLongConstant(0, getArrayElmtCount(arrayVarType));
            returnVector.add(Xcons.List(varId.Ref(), count, Xcons.SizeOf(((ArrayType)varType).getArrayElementType())));
          } break;
        default:
          throw new XMPexception("'" + varName + "' has a wrong data type for broadcast");
      }
    }

    return returnVector;
  }

  private Block createBcastFuncCallBlock(boolean isMacro, String funcType, Xobject execDesc, Vector<XobjList> funcArgsList,
                                         XMPpair<String, XobjList> execFromRefArgs) throws XMPexception {
    String funcSurfix = null;
    XobjList fromRef = null;
    if (execFromRefArgs == null) funcSurfix = new String(funcType + "_OMITTED");
    else {
      funcSurfix = new String(funcType + "_" + execFromRefArgs.getFirst());
      fromRef = execFromRefArgs.getSecond();
    }

    Ident funcId = null;
    if (isMacro) funcId = XMP.getMacroId("_XCALABLEMP_M_BCAST_" + funcSurfix);
    else         funcId = _globalDecl.declExternFunc("_XCALABLEMP_bcast_" + funcSurfix);

    BlockList funcCallList = Bcons.emptyBody();
    Iterator<XobjList> it = funcArgsList.iterator();
    while (it.hasNext()) {
      XobjList funcArgs = it.next();
      if (execDesc != null) funcArgs.cons(execDesc);
      if (execFromRefArgs != null) XMPutil.mergeLists(funcArgs, fromRef);

      funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
    }
    return Bcons.COMPOUND(funcCallList);
  }

  private XMPpair<String, XobjList> createExecFromRefArgs(XobjList fromRef,
                                                          XMPobjectTable localObjectTable) throws XMPexception {
    if (fromRef.getArg(0) == null) {
      // execute on global communicator
      XobjList globalRef = (XobjList)fromRef.getArg(1);

      XobjList execFuncArgs = Xcons.List();
      // lower
      if (globalRef.getArg(0) == null)
        throw new XMPexception("lower bound cannot be omitted in <from-ref>");
      else execFuncArgs.add(globalRef.getArg(0));

      // upper
      if (globalRef.getArg(1) == null)
        throw new XMPexception("upper bound cannot be omitted in <from-ref>");
      else execFuncArgs.add(globalRef.getArg(1));

      // stride
      if (globalRef.getArg(2) == null) execFuncArgs.add(Xcons.IntConstant(1));
      else execFuncArgs.add(globalRef.getArg(2));

      return new XMPpair<String, XobjList>(new String("GLOBAL"), execFuncArgs);
    }
    else {
      // execute on <object-ref>

      // check object name collision
      String objectName = fromRef.getArg(0).getString();
      XMPobject fromRefObject = getXMPobject(objectName, localObjectTable);
      if (fromRefObject.getKind() == XMPobject.TEMPLATE)
        throw new XMPexception("template cannot be used in <from-ref>");

      // create arguments
      if (fromRef.getArg(1) == null)
        throw new XMPexception("multiple source nodes indicated in bcast directive");
      else {
        XobjList execFuncArgs = Xcons.List(fromRefObject.getDescId().Ref());

        int refIndex = 0;
        int refDim = fromRefObject.getDim();
        for (XobjArgs i = fromRef.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
          if (refIndex == refDim)
            throw new XMPexception("wrong nodes dimension indicated, too many");

          XobjList t = (XobjList)i.getArg();
          if (t == null) execFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
          else {
            execFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(0)));

            // lower
            if (t.getArg(0) == null)
              throw new XMPexception("lower bound cannot be omitted in <from-ref>");
            else execFuncArgs.add(Xcons.Cast(Xtype.intType, t.getArg(0)));

            // upper
            if (t.getArg(1) == null)
              throw new XMPexception("upper bound cannot be omitted in <from-ref>");
            else execFuncArgs.add(Xcons.Cast(Xtype.intType, t.getArg(1)));

            // stride
            if (t.getArg(2) == null) execFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
            else execFuncArgs.add(Xcons.Cast(Xtype.intType, t.getArg(2)));
          }

          refIndex++;
        }

        if (refIndex != refDim)
          throw new XMPexception("the number of <nodes/template-subscript> should be the same with the dimension");

        return new XMPpair<String, XobjList>(new String("NODES"), execFuncArgs);
      }
    }
  }

  private void translateGmove(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList gmoveDecl = (XobjList)pb.getClauses();
    XMPobjectTable localObjectTable = XMPlocalDecl.declObjectTable(pb);
    BlockList gmoveBody = pb.getBody();

    // check body
    Xobject assignStmt = null;
    String checkBodyErrMsg = new String("gmove directive should be written before one assign statement");
    Block gmoveBodyHead = gmoveBody.getHead();
    if(gmoveBodyHead instanceof SimpleBlock) {
      if (gmoveBodyHead.getNext() != null)
        throw new XMPexception(checkBodyErrMsg);

      Statement gmoveStmt = gmoveBodyHead.getBasicBlock().getHead();
      if (gmoveStmt.getNext() != null)
        throw new XMPexception(checkBodyErrMsg);

      if(gmoveStmt.getExpr().Opcode() == Xcode.ASSIGN_EXPR)
        assignStmt = gmoveStmt.getExpr();
      else
        throw new XMPexception(checkBodyErrMsg);
    }
    else
      throw new XMPexception(checkBodyErrMsg);

    // FIXME consider in, out clause
    XMPpair<XMPalignedArray, XobjList> leftExpr = getAlignedArrayExpr(assignStmt.left(), localObjectTable);
    XMPpair<XMPalignedArray, XobjList> rightExpr = getAlignedArrayExpr(assignStmt.right(), localObjectTable);
    if (leftExpr == null) {
      if (rightExpr == null) {		// !leftIsAlignedArray && !rightIsAlignedArray	|-> local assignment (every node)
        pb.replace(Bcons.COMPOUND(gmoveBody));
      }
      else {				// !leftIsAlignedArray &&  rightIsAlignedArray	|-> broadcast
        XMPalignedArray alignedArray = rightExpr.getFirst();

        // XXX left/right is not a constant
        XobjList gmoveFuncArgs = Xcons.List(Xcons.AddrOf(assignStmt.left()), Xcons.AddrOf(assignStmt.right()),
                                            Xcons.SizeOf(alignedArray.getType()),
                                            alignedArray.getDescId().Ref());
        XMPutil.mergeLists(gmoveFuncArgs, rightExpr.getSecond());

        pb.replace(createFuncCallBlock("_XCALABLEMP_gmove_BCAST_SCALAR", gmoveFuncArgs));
      }
    }
    else {
      if (rightExpr == null) {		//  leftIsAlignedArray && !rightIsAlignedArray	|-> local assignment (home node)
        XMPalignedArray alignedArray = leftExpr.getFirst();

        XobjList gmoveFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
        XMPutil.mergeLists(gmoveFuncArgs, leftExpr.getSecond());

        Ident gmoveFuncId = _env.declExternIdent("_XCALABLEMP_gmove_exec_home_SCALAR", Xtype.Function(Xtype.boolType));
        pb.replace(Bcons.IF(BasicBlock.Cond(gmoveFuncId.Call(gmoveFuncArgs)),
                            gmoveBody, null));
      }
      else {				//  leftIsAlignedArray &&  rightIsAlignedArray	|-> send/recv
        XMPalignedArray leftAlignedArray = leftExpr.getFirst();
        XMPalignedArray rightAlignedArray = rightExpr.getFirst();

        // FIXME type check: left right is the same -> needed???

        // XXX left/right is not a constant
        XobjList gmoveFuncArgs = Xcons.List(Xcons.AddrOf(assignStmt.left()), Xcons.AddrOf(assignStmt.right()),
                                            Xcons.SizeOf(leftAlignedArray.getType()),
                                            leftAlignedArray.getDescId().Ref(), rightAlignedArray.getDescId().Ref());
        XMPutil.mergeLists(gmoveFuncArgs, leftExpr.getSecond());
        XMPutil.mergeLists(gmoveFuncArgs, rightExpr.getSecond());

        pb.replace(createFuncCallBlock("_XCALABLEMP_gmove_SENDRECV_SCALAR", gmoveFuncArgs));
      }
    }
  }

  public XMPpair<XMPalignedArray, XobjList> getAlignedArrayExpr(Xobject expr, XMPobjectTable localObjectTable) throws XMPexception {
    if (expr == null) return null;

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject newExpr = null;
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) continue;

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            String arrayName = myExpr.getSym();
            XMPalignedArray alignedArray = findXMPalignedArray(arrayName, localObjectTable);
            if (alignedArray != null) {
              iter.next();
              return new XMPpair<XMPalignedArray, XobjList>(alignedArray,
                                                            parseAlignedArrayExpr(iter, alignedArray, 0, Xcons.List()));
            }

            break;
          }
        default:
          break;
      }
    }

    return null;
  }

  private XobjList parseAlignedArrayExpr(bottomupXobjectIterator iter,
                                         XMPalignedArray alignedArray, int arrayDimCount, XobjList arrayRefs) throws XMPexception {
    String syntaxErrMsg = "syntax error on array expression, expression is not appropriate for gmove directive";
    Xcode prevExprOpcode = iter.getPrevXobject().Opcode();
    Xobject myExpr = iter.getXobject();
    switch (myExpr.Opcode()) {
      case PLUS_EXPR:
        {
          switch (prevExprOpcode) {
            case ARRAY_REF:
              {
                if (arrayDimCount != 0)
                  throw new XMPexception(syntaxErrMsg);

                break;
              }
            case POINTER_REF:
              break;
            default:
              throw new XMPexception(syntaxErrMsg);
          }

          arrayRefs.add(Xcons.Cast(Xtype.intType, myExpr.right()));
          iter.next();
          return parseAlignedArrayExpr(iter, alignedArray, arrayDimCount + 1, arrayRefs);
        }
      case POINTER_REF:
        {
          switch (prevExprOpcode) {
            case PLUS_EXPR:
              break;
            default:
              throw new XMPexception(syntaxErrMsg);
          }

          iter.next();
          if (iter.end()) {
            if (alignedArray.getDim() == arrayDimCount) return arrayRefs;
            else
              throw new XMPexception(syntaxErrMsg);
          }
          else
            return parseAlignedArrayExpr(iter, alignedArray, arrayDimCount, arrayRefs);
        }
      default:
        throw new XMPexception(syntaxErrMsg);
    }
  }

  private XMPtriplet<String, Boolean, XobjList> createExecOnRefArgs(XobjList onRef,
                                                                    XMPobjectTable localObjectTable) throws XMPexception {
    if (onRef.getArg(0) == null) {
      // execute on global communicator
      XobjList globalRef = (XobjList)onRef.getArg(1);

      boolean splitComm = false;
      XobjList tempArgs = Xcons.List();
      // lower
      if (globalRef.getArg(0) == null) tempArgs.add(Xcons.IntConstant(1));
      else {
        splitComm = true;
        tempArgs.add(globalRef.getArg(0));
      }
      // upper
      if (globalRef.getArg(1) == null) tempArgs.add(_globalDecl.getWorldSizeId().Ref());
      else {
        splitComm = true;
        tempArgs.add(globalRef.getArg(1));
      }
      // stride
      if (globalRef.getArg(2) == null) tempArgs.add(Xcons.IntConstant(1));
      else {
        splitComm = true;
        tempArgs.add(globalRef.getArg(2));
      }

      String execFuncSurfix = null;
      XobjList execFuncArgs = null;
      if (splitComm) {
        execFuncSurfix = "GLOBAL_PART";
        execFuncArgs = tempArgs;
      }
      else {
        execFuncSurfix = "NODES_ENTIRE";
        execFuncArgs = Xcons.List(_globalDecl.getWorldDescId().Ref());
      }

      return new XMPtriplet<String, Boolean, XobjList>(execFuncSurfix, new Boolean(splitComm), execFuncArgs);
    }
    else {
      // execute on <object-ref>

      // check object name collision
      String objectName = onRef.getArg(0).getString();
      XMPobject onRefObject = getXMPobject(objectName, localObjectTable);

      Xobject ontoNodesRef = null;
      Xtype castType = null;
      switch (onRefObject.getKind()) {
        case XMPobject.NODES:
          ontoNodesRef = onRefObject.getDescId().Ref();
          castType = Xtype.intType;
          break;
        case XMPobject.TEMPLATE:
          {
            XMPnodes ontoNodes = ((XMPtemplate)onRefObject).getOntoNodes();
            if (ontoNodes == null)
              throw new XMPexception("template '" + objectName + "' is not distributed");

            ontoNodesRef = ontoNodes.getDescId().Ref();
            castType = Xtype.longlongType;
            break;
          }
        default:
          throw new XMPexception("unknown object type");
      }

      // create arguments
      if (onRef.getArg(1) == null)
        return new XMPtriplet<String, Boolean, XobjList>(new String("NODES_ENTIRE"), new Boolean(false), Xcons.List(ontoNodesRef));
      else {
        boolean splitComm = false;
        int refIndex = 0;
        int refDim = onRefObject.getDim();
        boolean getUpper = false;
        XobjList tempArgs = Xcons.List();
        for (XobjArgs i = onRef.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
          if (refIndex == refDim)
            throw new XMPexception("wrong nodes dimension indicated, too many");

          XobjList t = (XobjList)i.getArg();
          if (t == null) {
            splitComm = true;
            tempArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
          }
          else {
            tempArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(0)));

            // lower
            if (t.getArg(0) == null) tempArgs.add(Xcons.Cast(castType, Xcons.IntConstant(1)));
            else {
              splitComm = true;
              tempArgs.add(Xcons.Cast(castType, t.getArg(0)));
            }
            // upper
            if (t.getArg(1) == null) {
              Xobject onRefUpper = onRefObject.getUpperAt(refIndex);
              if (onRefUpper == null) getUpper = true;
              else tempArgs.add(Xcons.Cast(castType, onRefUpper));
            }
            else {
              splitComm = true;
              tempArgs.add(Xcons.Cast(castType, t.getArg(1)));
            }
            // stride
            if (t.getArg(2) == null) tempArgs.add(Xcons.Cast(castType, Xcons.IntConstant(1)));
            else {
              splitComm = true;
              tempArgs.add(Xcons.Cast(castType, t.getArg(2)));
            }
          }

          refIndex++;
        }

        if (refIndex != refDim)
          throw new XMPexception("the number of <nodes/template-subscript> should be the same with the dimension");

        if (splitComm) {
          String execFuncSurfix = null;
          XobjList execFuncArgs = null;
          execFuncArgs = tempArgs;
          switch (onRefObject.getKind()) {
            case XMPobject.NODES:
              execFuncSurfix = "NODES_PART";
              execFuncArgs.cons(ontoNodesRef);
              break;
            case XMPobject.TEMPLATE:
              execFuncSurfix = "TEMPLATE_PART";
              execFuncArgs.cons(((XMPtemplate)onRefObject).getDescId().Ref());
              break;
            default:
              throw new XMPexception("unknown object type");
          }

          if (getUpper) execFuncArgs.cons(Xcons.IntConstant(1));
          else          execFuncArgs.cons(Xcons.IntConstant(0));

          return new XMPtriplet<String, Boolean, XobjList>(execFuncSurfix, new Boolean(splitComm), execFuncArgs);
        }
        else
          return new XMPtriplet<String, Boolean, XobjList>(new String("NODES_ENTIRE"),
                                                           new Boolean(splitComm), Xcons.List(ontoNodesRef));
      }
    }
  }

  private void setupFinalizer(BlockList body, Ident funcId, XobjList args) throws XMPexception {
    BlockIterator i = new topdownBlockIterator(body);
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() == Xcode.GOTO_STATEMENT)
        throw new XMPexception("cannot use goto statement here");
      else if (b.Opcode() == Xcode.RETURN_STATEMENT)
        b.insert(funcId.Call(args));
    }
    body.add(funcId.Call(args));
  }

  public static void checkDeclPragmaLocation(PragmaBlock pb) throws XMPexception {
/*
    String pragmaNameL = pb.getPragma().toLowerCase();
    String errMsg = new String(pragmaNameL +
                               " directive should be written before declarations, statements and executable directives");

    // check parent
    Block parentBlock = pb.getParentBlock();
    if (parentBlock.Opcode() != Xcode.COMPOUND_STATEMENT)
      throw new XMPexception(errMsg);
    else {
      BlockList parent = pb.getParent();
      Xobject declList = parent.getDecls();
      if (declList != null) {
        if (declList.operand() != null)
          throw new XMPexception(errMsg);
      }

      if (parentBlock.getParentBlock().Opcode() != Xcode.FUNCTION_DEFINITION)
        throw new XMPexception(errMsg);
    }

    // check previous blocks
    for (Block prevBlock = pb.getPrev(); prevBlock != null; prevBlock = prevBlock.getPrev()) {
      if (prevBlock.Opcode() == Xcode.XMP_PRAGMA) {
        XMPpragma prevPragma = XMPpragma.valueOf(((PragmaBlock)prevBlock).getPragma());
        switch (XMPpragma.valueOf(((PragmaBlock)prevBlock).getPragma())) {
          case NODES:
          case TEMPLATE:
          case DISTRIBUTE:
          case ALIGN:
          case SHADOW:
            continue;
          default:
            throw new XMPexception(errMsg);
        }
      }
      else
        throw new XMPexception(errMsg);
    }
*/
    // XXX delete this
    return;
  }

  private XMPobject getXMPobject(String objectName, XMPobjectTable localObjectTable) throws XMPexception {
    XMPobject object = localObjectTable.getObject(objectName);
    if (object == null) {
      object = _globalObjectTable.getObject(objectName);
      if (object == null)
        throw new XMPexception("cannot find '" + objectName + "' nodes/template");
    }

    return object;
  }

  private XMPtemplate getXMPtemplate(String templateName, XMPobjectTable localObjectTable) throws XMPexception {
    XMPtemplate t = localObjectTable.getTemplate(templateName);
    if (t == null) {
      t = _globalObjectTable.getTemplate(templateName);
      if (t == null)
        throw new XMPexception("template '" + templateName + "' is not declared");
    }

    return t;
  }

  private XMPnodes getXMPnodes(String nodesName, XMPobjectTable localObjectTable) throws XMPexception {
    XMPnodes n = localObjectTable.getNodes(nodesName);
    if (n == null) {
      n = _globalObjectTable.getNodes(nodesName);
      if (n == null)
        throw new XMPexception("nodes '" + nodesName + "' is not declared");
    }

    return n;
  }

  private XMPalignedArray getXMPalignedArray(String arrayName, XMPobjectTable localObjectTable) throws XMPexception {
    XMPalignedArray a = localObjectTable.getAlignedArray(arrayName);
    if (a == null) {
      a = _globalObjectTable.getAlignedArray(arrayName);
      if (a == null)
        throw new XMPexception("array '" + arrayName + "' is not aligned");
    }

    return a;
  }

  private XMPalignedArray findXMPalignedArray(String arrayName, XMPobjectTable localObjectTable) throws XMPexception {
    XMPalignedArray a = localObjectTable.getAlignedArray(arrayName);
    if (a == null) {
      a = _globalObjectTable.getAlignedArray(arrayName);
    }

    return a;
  }
}
