/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.util.Vector;

public class XMPtemplate extends XMPobject {
  // defined in xmp_constant.h
  public final static int DUPLICATION	= 100;
  public final static int BLOCK		= 101;
  public final static int CYCLIC	= 102;

  private boolean		_isFixed;
  private boolean		_isDistributed;
  private XMPnodes		_ontoNodes;
  private Vector<XobjInt>	_ontoNodesIndexVector;
  private Vector<Integer>	_distMannerVector;
  private Vector<Xobject>	_lowerVector;

  public XMPtemplate(String name, int dim, Ident descId) {
    super(XMPobject.TEMPLATE, name, dim, descId);

    _isFixed = false;
    _isDistributed = false;
    _ontoNodes = null;
    _ontoNodesIndexVector = new Vector<XobjInt>();
    _distMannerVector = new Vector<Integer>();

    for (int i = 0; i < dim; i++) {
      _ontoNodesIndexVector.add(null);
      _distMannerVector.add(null);
    }

    _lowerVector = new Vector<Xobject>();
  }

  public void setIsFixed() {
    _isFixed = true;
  }

  public boolean isFixed() {
    return _isFixed;
  }

  public void setIsDistributed() {
    _isDistributed = true;
  }

  public boolean isDistributed() {
    return _isDistributed;
  }

  public void setOntoNodes(XMPnodes nodes) {
    _ontoNodes = nodes;
  }

  public XMPnodes getOntoNodes() {
    return _ontoNodes;
  }

  public void setOntoNodesIndexAt(int nodesDimIdx, int templateDimIdx) {
    _ontoNodesIndexVector.setElementAt(Xcons.IntConstant(nodesDimIdx), templateDimIdx);
  }

  public XobjInt getOntoNodesIndexAt(int index) {
    return _ontoNodesIndexVector.get(index);
  }

  public void setDistMannerAt(int manner, int index) {
    _distMannerVector.setElementAt(new Integer(manner), index);
  }

  public int getDistMannerAt(int index) throws XMPexception {
    if (!_isDistributed) {
      throw new XMPexception("template " + getName() + " is not distributed");
    }

    return _distMannerVector.get(index).intValue();
  }

  public String getDistMannerStringAt(int index) throws XMPexception {
    if (!_isDistributed) {
      throw new XMPexception("template " + getName() + " is not distributed");
    }

    switch (getDistMannerAt(index)) {
      case DUPLICATION:
        return new String("DUPLICATION");
      case BLOCK:
        return new String("BLOCK");
      case CYCLIC:
        return new String("CYCLIC");
      default:
        throw new XMPexception("unknown distribute manner");
    }
  }

  public void addLower(Xobject lower) {
    _lowerVector.add(lower);
  }

  public Xobject getLowerAt(int index) {
    return _lowerVector.get(index);
  }

  public static String getDistMannerString(int manner) throws XMPexception {
    switch (manner) {
      case DUPLICATION:
        return new String("DUPLICATION");
      case BLOCK:
        return new String("BLOCK");
      case CYCLIC:
        return new String("CYCLIC");
      default:
        throw new XMPexception("unknown distribute manner");
    }
  }

  @Override
  public boolean checkInheritExec() {
    return _ontoNodes.checkInheritExec();
  }

  public static void translateTemplate(XobjList templateDecl, XMPglobalDecl globalDecl,
                                       boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // local parameters
    BlockList funcBlockList = null;
    XMPsymbolTable localXMPsymbolTable = null;
    if (isLocalPragma) {
      funcBlockList = XMPlocalDecl.findParentFunctionBlock(pb).getBody();
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    }

    // check name collision
    String templateName = templateDecl.getArg(0).getString();
    if (isLocalPragma) {
      XMPlocalDecl.checkObjectNameCollision(templateName, funcBlockList, localXMPsymbolTable);
    }
    else {
      globalDecl.checkObjectNameCollision(templateName);
    }

    // declare template desciptor
    Ident templateDescId = null;
    if (isLocalPragma) {
      templateDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + templateName, pb);
    }
    else {
      templateDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + templateName, Xtype.voidPtrType);
    }

    // declare template object
    int templateDim = 0;
    for (XobjArgs i = templateDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) templateDim++;
    if (templateDim > XMP.MAX_DIM) {
      throw new XMPexception("template dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    XMPtemplate templateObject = new XMPtemplate(templateName, templateDim, templateDescId);
    if (isLocalPragma) {
      localXMPsymbolTable.putXMPobject(templateObject);
    }
    else {
      globalDecl.putXMPobject(templateObject);
    }

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

    String constructorName = new String("_XMP_init_template_");
    if (templateIsFixed) {
      templateObject.setIsFixed();
      constructorName += "FIXED";
    }
    else {
      constructorName += "UNFIXED";
    }

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall(constructorName, templateArgs, globalDecl, pb);
      XMPlocalDecl.insertDestructorCall("_XMP_finalize_template", Xcons.List(templateDescId.Ref()), globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall(constructorName, templateArgs);
    }
  }

  public static void translateDistribute(XobjList distDecl, XMPglobalDecl globalDecl,
                                         boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // local parameters
    BlockList funcBlockList = null;
    XMPsymbolTable localXMPsymbolTable = null;
    if (isLocalPragma) {
      funcBlockList = XMPlocalDecl.findParentFunctionBlock(pb).getBody();
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    }

    // get template object
    String templateName = distDecl.getArg(0).getString();
    XMPtemplate templateObject = null;
    if (isLocalPragma) {
      templateObject = localXMPsymbolTable.getXMPtemplate(templateName);
      if (templateObject == null) {
        templateObject = globalDecl.getXMPtemplate(templateName);
        if (templateObject != null) {
          throw new XMPexception("global template cannot be distributed in local scope");
        }
      }
    }
    else {
      templateObject = globalDecl.getXMPtemplate(templateName);
    }

    if (templateObject == null) {
      throw new XMPexception("template '" + templateName + "' is not declared");
    }

    if (!templateObject.isFixed()) {
      throw new XMPexception("template '" + templateName + "' is not fixed");
    }

    if (templateObject.isDistributed()) {
      throw new XMPexception("template '" + templateName + "' is already distributed");
    }

    // get nodes object
    String nodesName = distDecl.getArg(2).getString();
    XMPnodes nodesObject = null;
    if (isLocalPragma) {
      nodesObject = XMPlocalDecl.getXMPnodes(nodesName, localXMPsymbolTable, globalDecl);
    }
    else {
      nodesObject = globalDecl.getXMPnodes(nodesName);
    }

    if (nodesObject == null) {
      throw new XMPexception("nodes '" + nodesName + "' is not declared");
    }

    templateObject.setOntoNodes(nodesObject);

    // setup chunk constructor
    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall("_XMP_init_template_chunk",
                                      Xcons.List(templateObject.getDescId().Ref(),
                                                 nodesObject.getDescId().Ref()),
                                      globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_init_template_chunk",
                                        Xcons.List(templateObject.getDescId().Ref(),
                                                   nodesObject.getDescId().Ref()));
    }

    // create distribute function calls
    int templateDim = templateObject.getDim();
    int templateDimIdx = 0;
    int nodesDim = nodesObject.getDim();
    int nodesDimIdx = 0;
    for (XobjArgs i = distDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      if (templateDimIdx == templateDim) {
        throw new XMPexception("wrong template dimension indicated, too many");
      }

      XobjList distManner = (XobjList)i.getArg();
      setupDistribution(distManner, templateObject, templateDimIdx, nodesDimIdx, globalDecl, isLocalPragma, pb);

      int distMannerValue = distManner.getArg(0).getInt();
      // FIXME support gblock
      switch (distMannerValue) {
        case XMPtemplate.BLOCK:
        case XMPtemplate.CYCLIC:
          {
            if (nodesDimIdx == nodesDim) {
              throw new XMPexception("the number of <dist-format> (except '*') should be the same with the nodes dimension");
            }

            nodesDimIdx++;
            break;
          }
        default:
      }

      templateDimIdx++;
    }

    // check nodes, template dimension
    if (nodesDimIdx != nodesDim) {
      throw new XMPexception("the number of <dist-format> (except '*') should be the same with the nodes dimension");
    }

    if (templateDimIdx != templateDim) {
      throw new XMPexception("wrong template dimension indicated, too few");
    }

    // set distributed
    templateObject.setIsDistributed();
  }

  private static void setupDistribution(XobjList distManner, XMPtemplate templateObject,
                                        int templateDimIdx, int nodesDimIdx,
                                        XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    XobjList funcArgs = null;
    int distMannerValue = distManner.getArg(0).getInt();
    String distMannerName = null;
    switch (distMannerValue) {
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
          Xobject distMannerWidth = distManner.getArg(1);
          if (distMannerWidth == null) {
            distMannerName = "CYCLIC";
            funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                  Xcons.IntConstant(templateDimIdx),
                                  Xcons.IntConstant(nodesDimIdx));
            templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
          } else {
            throw new XMPexception("cyclic(w) is not supported yet");
          }

          break;
        }
      default:
        throw new XMPexception("unknown distribute manner");
    }

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall("_XMP_dist_template_" + distMannerName, funcArgs, globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_dist_template_" + distMannerName, funcArgs);
    }

    templateObject.setDistMannerAt(distMannerValue, templateDimIdx);
  }
}
