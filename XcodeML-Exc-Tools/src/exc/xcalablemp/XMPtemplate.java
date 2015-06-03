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
  public final static int BLOCK_CYCLIC	= 103;
  public final static int GBLOCK	= 104;

  private boolean		_isFixed;
  private boolean		_isDistributed;
  private boolean               _distributionIsFixed;
  private XMPnodes		_ontoNodes;
  private Vector<XobjInt>	_ontoNodesIndexVector;
  private Vector<Integer>	_distMannerVector;
  private Vector<Xobject>       _sizeVector;
  private Vector<Xobject>       _widthVector;
  private XobjList              _decl;
  private XobjList              _distDecl;
  private Vector<Ident>		_gtolTemp0IdVector;

  private boolean               _isStaticDesc = false;
  private Ident                 _flagId = null;

  public XMPtemplate(String name, int dim, Ident descId) {
    super(XMPobject.TEMPLATE, name, dim, descId);

    _isFixed = false;
    _isDistributed = false;
    _distributionIsFixed = true;
    _ontoNodes = null;
    _ontoNodesIndexVector = new Vector<XobjInt>();
    _distMannerVector = new Vector<Integer>();
    _sizeVector = new Vector<Xobject>();
    _widthVector = new Vector<Xobject>();
    _gtolTemp0IdVector = new Vector<Ident>();

    for (int i = 0; i < dim; i++) {
      _ontoNodesIndexVector.add(null);
      _distMannerVector.add(null);
      _widthVector.add(null);
      _gtolTemp0IdVector.add(null);
    }
    
    _decl = null;
    _distDecl = null;
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

  public void setDistributionIsFixed(boolean b) {
    _distributionIsFixed = b;
  }

  public void setDistributionIsFixed() {
    _distributionIsFixed = true;
  }

  public boolean distributionIsFixed() {
    return _distributionIsFixed;
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
    } else {
      return getDistMannerString(getDistMannerAt(index));
    }
  }

  public void createSizeVector() {
    for (int i = 0; i < this.getDim(); i++) {
      _sizeVector.add(Xcons.binaryOp(Xcode.PLUS_EXPR,
                                     Xcons.binaryOp(Xcode.MINUS_EXPR,
                                                    this.getUpperAt(i),
                                                    this.getLowerAt(i)),
                                     Xcons.IntConstant(1)));
    }
  }

  public Xobject getSizeAt(int index) {
    return _sizeVector.get(index);
  }

  public void setWidthAt(Xobject width, int index) {
    _widthVector.setElementAt(width, index);
  }

  public Xobject getWidthAt(int index) throws XMPexception {
    return _widthVector.get(index);
  }

  public static String getDistMannerString(int manner) throws XMPexception {
    switch (manner) {
      case DUPLICATION:
        return new String("DUPLICATION");
      case BLOCK:
        return new String("BLOCK");
      case CYCLIC:
        return new String("CYCLIC");
      case BLOCK_CYCLIC:
        return new String("BLOCK_CYCLIC");
      case GBLOCK:
        return new String("GBLOCK");
      default:
        throw new XMPexception("unknown distribute manner");
    }
  }

  public void setDecl(XobjList decl){
    _decl = decl;
  }

  public XobjList getDecl(){
    return _decl;
  }

  public void setDistDecl(XobjList distDecl){
    _distDecl = distDecl;
  }

  public XobjList getDistDecl(){
    return _distDecl;
  }

  public void setGtolTemp0IdAt(Ident temp0Id, int index) {
    _gtolTemp0IdVector.setElementAt(temp0Id, index);
  }

  public Ident getGtolTemp0IdAt(int index) {
    return _gtolTemp0IdVector.get(index);
  }

  public void setIsStaticDesc(boolean flag){
    _isStaticDesc = flag;
  }

  public boolean isStaticDesc(){
    return _isStaticDesc;
  }

  public void setFlagId(Ident id){
    _flagId = id;
  }

  public Ident getFlagId(){
    return _flagId;
  }

  @Override
  public boolean checkInheritExec() {
    return _ontoNodes.checkInheritExec();
  }

  public static void translateTemplate(XobjList templateDecl, XMPglobalDecl globalDecl,
                                       boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // local parameters
    //BlockList funcBlockList = null;
    XMPsymbolTable localXMPsymbolTable = null;
    Block parentBlock = null;
    boolean isStaticDesc = false;

    if (isLocalPragma) {
      //funcBlockList = XMPlocalDecl.findParentFunctionBlock(pb).getBody();
      //localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
      parentBlock = pb.getParentBlock();
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable2(parentBlock);
    }

    // check name collision
    String templateName = templateDecl.getArg(0).getString();
    if (isLocalPragma) {
      //XMPlocalDecl.checkObjectNameCollision(templateName, funcBlockList, localXMPsymbolTable);
      XMPlocalDecl.checkObjectNameCollision(templateName, parentBlock.getBody(), localXMPsymbolTable);
      isStaticDesc = localXMPsymbolTable.isStaticDesc(templateName);
    }
    else {
      globalDecl.checkObjectNameCollision(templateName);
    }

    // declare template desciptor
    Ident templateDescId = null;
    if (isLocalPragma) {
      //templateDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + templateName, pb);
      templateDescId = XMPlocalDecl.addObjectId2(XMP.DESC_PREFIX_ + templateName, parentBlock);
    }
    else {
      templateDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + templateName, Xtype.voidPtrType);
    }

    if (isStaticDesc) templateDescId.setStorageClass(StorageClass.STATIC);

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

    templateObject.setDecl(templateDecl);

    // create function call
    boolean templateIsFixed = true;
    XobjList templateArgs = Xcons.List(templateDescId.getAddr(), Xcons.IntConstant(templateDim));
    for (XobjArgs i = templateDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      Xobject templateSpec = i.getArg();
      if (templateSpec == null || (templateSpec instanceof XobjList && templateSpec.Nargs() == 0)) {
        templateIsFixed = false;
        //throw new XMPexception("template should has fixed size in this version");
      } else {

	if (!templateIsFixed)
	  throw new XMPexception("Every <template-spec> shall be either [int-expr :] int-expr or ':'");

        Xobject templateLower = templateSpec.left();
        Xobject templateUpper = templateSpec.right();

        templateArgs.add(Xcons.Cast(Xtype.longlongType, templateLower));
        templateArgs.add(Xcons.Cast(Xtype.longlongType, templateUpper));

        templateObject.addLower(templateLower);
        templateObject.addUpper(templateUpper);
      }
    }

    // check static_desc
    if (isLocalPragma) templateObject.setIsStaticDesc(isStaticDesc);
    if (!templateIsFixed && isStaticDesc)
      throw new XMPexception("non-fixed template cannot have the static_desc attribute.");

    if (templateIsFixed)
      templateObject.createSizeVector();

    String constructorName = new String("_XMP_init_template_");
    if (templateIsFixed) {
      templateObject.setIsFixed();
      constructorName += "FIXED";
    } else {
      constructorName += "UNFIXED";
    }

    if (isLocalPragma) {

      if (isStaticDesc){
	Ident id = parentBlock.getBody().declLocalIdent(XMP.STATIC_DESC_PREFIX_ + templateName, Xtype.intType,
							StorageClass.STATIC, Xcons.IntConstant(0));
	templateObject.setFlagId(id);
	XMPlocalDecl.addConstructorCall2_staticDesc(constructorName, templateArgs, globalDecl, parentBlock, id, false);
      }
      else {
	XMPlocalDecl.addConstructorCall2(constructorName, templateArgs, globalDecl, parentBlock);
      }

      if (!isStaticDesc)
	XMPlocalDecl.insertDestructorCall2("_XMP_finalize_template", Xcons.List(templateDescId.Ref()), globalDecl, parentBlock);

    } else {
      globalDecl.addGlobalInitFuncCall(constructorName, templateArgs);
    }
  }

  public static void translateDistribute(XobjList distDecl, XMPglobalDecl globalDecl,
                                         boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // local parameters
    //BlockList funcBlockList = null;
    XMPsymbolTable localXMPsymbolTable = null;
    Block parentBlock = null;
    if (isLocalPragma) {
      //funcBlockList = XMPlocalDecl.findParentFunctionBlock(pb).getBody();
      //localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
      parentBlock = pb.getParentBlock();
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable2(parentBlock);
    }

    // get template object
    String templateName = distDecl.getArg(0).getString();
    XMPtemplate templateObject = null;
    // if (isLocalPragma) {
    //   templateObject = localXMPsymbolTable.getXMPtemplate(templateName);
    //   if (templateObject == null) {
    //     templateObject = globalDecl.getXMPtemplate(templateName);
    //     if (templateObject != null) {
    //       throw new XMPexception("global template cannot be distributed in local scope");
    //     }
    //   }
    // }
    // else {
    //   templateObject = globalDecl.getXMPtemplate(templateName);
    // }
    templateObject = globalDecl.getXMPtemplate(templateName, pb);
    if (templateObject == null) {
      throw new XMPexception("template '" + templateName + "' is not declared");
    }

    templateObject.setDistDecl(distDecl);

    // if (!templateObject.isFixed()) {
    //   throw new XMPexception("template '" + templateName + "' is not fixed");
    // }

    if (templateObject.isDistributed()) {
       throw new XMPexception("template '" + templateName + "' is already distributed");
    }

    // get nodes object
    String nodesName = distDecl.getArg(2).getString();
    XMPnodes nodesObject = null;
    //nodesObject = globalDecl.getXMPnodes(nodesName, localXMPsymbolTable);
    nodesObject = globalDecl.getXMPnodes(nodesName, pb);
    if (nodesObject == null) {
      throw new XMPexception("nodes '" + nodesName + "' is not declared");
    }

    templateObject.setOntoNodes(nodesObject);

    if (templateObject.isFixed()){

      // setup chunk constructor
      if (isLocalPragma) {
	if (templateObject.isStaticDesc()){
	  XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_init_template_chunk",
						      Xcons.List(templateObject.getDescId().Ref(),
								 nodesObject.getDescId().Ref()),
						      globalDecl, parentBlock, templateObject.getFlagId(), false);
	}
	else {
	  XMPlocalDecl.addConstructorCall2("_XMP_init_template_chunk",
					   Xcons.List(templateObject.getDescId().Ref(),
						      nodesObject.getDescId().Ref()),
					   globalDecl, parentBlock);
	}
      }
      else {
	globalDecl.addGlobalInitFuncCall("_XMP_init_template_chunk",
					 Xcons.List(templateObject.getDescId().Ref(),
						    nodesObject.getDescId().Ref()));
      }

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
      setupDistribution(distManner, templateObject, templateDimIdx, nodesDimIdx, globalDecl, isLocalPragma, false, pb);

      int distMannerValue = distManner.getArg(0).getInt();
      switch (distMannerValue) {
        case XMPtemplate.BLOCK:
        case XMPtemplate.CYCLIC:
        case XMPtemplate.BLOCK_CYCLIC:
        case XMPtemplate.GBLOCK:
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

  private static Block setupDistribution(XobjList distManner, XMPtemplate templateObject,
					 int templateDimIdx, int nodesDimIdx,
					 XMPglobalDecl globalDecl, boolean isLocalPragma, boolean isTFIX,
					 PragmaBlock pb) throws XMPexception {

    XobjList funcArgs = null;
    int distMannerValue = distManner.getArg(0).getInt();
    String distMannerName = getDistMannerString(distMannerValue);
    switch (distMannerValue) {
      case XMPtemplate.DUPLICATION:
        {
          funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                Xcons.IntConstant(templateDimIdx));
          break;
        }
      case XMPtemplate.BLOCK:
        {
          funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                Xcons.IntConstant(templateDimIdx),
                                Xcons.IntConstant(nodesDimIdx));
          templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
          break;
        }
      case XMPtemplate.CYCLIC:
        {
          funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                Xcons.IntConstant(templateDimIdx),
                                Xcons.IntConstant(nodesDimIdx));
          templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
          break;
        }
      case XMPtemplate.BLOCK_CYCLIC:
        {
          Xobject width = distManner.getArg(1);
          funcArgs = Xcons.List(templateObject.getDescId().Ref(),
                                Xcons.IntConstant(templateDimIdx),
                                Xcons.IntConstant(nodesDimIdx),
                                width);
          templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
          templateObject.setWidthAt(width, templateDimIdx);
          break;
        }
      case XMPtemplate.GBLOCK:
        {
          Xobject mappingArray = distManner.getArg(1);
	  Xobject mappingArrayArg = mappingArray;
	  if (mappingArray instanceof XobjList && mappingArray.Nargs() == 0){
	    mappingArrayArg = Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0));
	    templateObject.setDistributionIsFixed(false);
	  }

	  if (templateObject.isFixed() || isTFIX){
	    Ident gtolTemp0Id = null;
	    String tempName = XMP.GTOL_PREFIX_ + "temp0_" + templateObject.getName() + "_" + templateDimIdx;
	    if (isLocalPragma) {
	      Block parentBlock = pb.getParentBlock();
	      gtolTemp0Id = XMPlocalDecl.addObjectId2(tempName, Xtype.intType, parentBlock);
	      if (templateObject.isStaticDesc()) gtolTemp0Id.setStorageClass(StorageClass.STATIC);
	    }
	    else {
	      gtolTemp0Id = globalDecl.declStaticIdent(tempName, Xtype.intType);
	    }

	    templateObject.setGtolTemp0IdAt(gtolTemp0Id, templateDimIdx);

	    funcArgs = Xcons.List(templateObject.getDescId().Ref(),
				  Xcons.IntConstant(templateDimIdx),
				  Xcons.IntConstant(nodesDimIdx),
				  mappingArrayArg, gtolTemp0Id.getAddr());
	  }

          templateObject.setOntoNodesIndexAt(nodesDimIdx, templateDimIdx);
	  templateObject.setWidthAt(mappingArray, templateDimIdx);
          break;
        }
      default:
        throw new XMPexception("unknown distribute manner");
    }

    if (templateObject.isFixed()){

      if (isLocalPragma) {
	Block parentBlock = pb.getParentBlock();
	if (templateObject.isStaticDesc()){
	  XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_dist_template_" + distMannerName,
						      funcArgs, globalDecl, parentBlock, templateObject.getFlagId(),
						      templateDimIdx == templateObject.getDim() - 1);
	}
	else {
	  XMPlocalDecl.addConstructorCall2("_XMP_dist_template_" + distMannerName,
					   funcArgs, globalDecl, parentBlock);
	}
      }
      else {
	globalDecl.addGlobalInitFuncCall("_XMP_dist_template_" + distMannerName, funcArgs);
      }

    }
    else if (isTFIX){
      Ident funcId = globalDecl.declExternFunc("_XMP_dist_template_" + distMannerName);
      return Bcons.Statement(funcId.Call(funcArgs));
    }

    templateObject.setDistMannerAt(distMannerValue, templateDimIdx);

    return null;
  }


  public static void translateTemplateFix(XobjList tfixDecl, XMPglobalDecl globalDecl,
					  PragmaBlock pb) throws XMPexception {

    BlockList tFixFuncBody = Bcons.emptyBody();

    Block parentBlock = pb.getParentBlock();

    String tName = tfixDecl.getArg(1).getString();

    XMPtemplate tObject = globalDecl.getXMPtemplate(tName, pb);

    XobjArgs i, j;

    if (tObject == null) {
      throw new XMPexception("template '" + tName + "' is not declared");
    }

    if (tObject.isFixed() && tObject.distributionIsFixed()) {
      throw new XMPexception("template '" + tName + "' is already fixed");
    }

    XobjArgs sizeArgs_decl = tObject.getDecl().getArg(1).getArgs();
    XobjArgs sizeArgs_tfix = tfixDecl.getArg(2).getArgs();

    if (tObject.getDistDecl() == null){
      throw new XMPexception("template '" + tName + "' is not distributed");
    }

    XobjArgs distMannerArgs_decl = tObject.getDistDecl().getArg(1).getArgs();
    XobjArgs distMannerArgs_tfix = tfixDecl.getArg(0).getArgs();

    if (!tObject.isFixed() && sizeArgs_tfix == null)
    	throw new XMPexception("No <template-spec> specified for a non-fixed template '" + tName + "'");

    if (!tObject.distributionIsFixed() && distMannerArgs_tfix == null)
    	throw new XMPexception("No <dist-format> specified for a non-fixed template '" + tName + "'");

    int tDim = tObject.getDim();

    // get nodes object
    XMPnodes nObject = tObject.getOntoNodes();

    //
    // create _XMP_set_template_size
    //

    // check rank matching
    for (i = sizeArgs_decl, j = sizeArgs_tfix;
	 i != null || j != null;
	 i = i.nextArgs(), j = j.nextArgs()){
      if (i == null || j == null){
	throw new XMPexception("the number of <template-spec> is different from that in the declaration");
      }
    }

    XobjArgs sizeArgs = tObject.isFixed() ? sizeArgs_decl : sizeArgs_tfix;

    XobjList tArgs = Xcons.List(tObject.getDescId().Ref(), Xcons.IntConstant(tDim));

    for (i = sizeArgs; i != null; i = i.nextArgs()) {
      Xobject tSpec = i.getArg();
      Xobject tLower = tSpec.left();
      Xobject tUpper = tSpec.right();

      tArgs.add(Xcons.Cast(Xtype.longlongType, tLower));
      tArgs.add(Xcons.Cast(Xtype.longlongType, tUpper));

      tObject.addLower(tLower);
      tObject.addUpper(tUpper);
    }

    tObject.createSizeVector();

    Ident funcId;

    //String constructorName = new String("_XMP_set_template_size");
    //XMPlocalDecl.addConstructorCall2(constructorName, tArgs, globalDecl, parentBlock);
    funcId = globalDecl.declExternFunc("_XMP_set_template_size");
    tFixFuncBody.add(Bcons.Statement(funcId.Call(tArgs)));

    //
    // create _XMP_init_template_chunk
    //
      
    // XMPlocalDecl.addConstructorCall2("_XMP_init_template_chunk",
    // 				     Xcons.List(tObject.getDescId().Ref(),
    // 						nObject.getDescId().Ref()),
    // 				     globalDecl, parentBlock);
    funcId = globalDecl.declExternFunc("_XMP_init_template_chunk");
    tFixFuncBody.add(Bcons.Statement(funcId.Call(Xcons.List(tObject.getDescId().Ref(),
    							    nObject.getDescId().Ref()))));

    //
    // create _XMP_dist_template_XXX
    //

    // check dist-format matching
    for (i = distMannerArgs_decl, j = distMannerArgs_tfix;
	 i != null && j != null;
	 i = i.nextArgs(), j = j.nextArgs()){

      XobjList distManner_decl = (XobjList)i.getArg();
      int distMannerValue_decl = distManner_decl.getArg(0).getInt();
      Xobject width_decl = distManner_decl.getArg(1);

      XobjList distManner_tfix = (XobjList)j.getArg();
      int distMannerValue_tfix = distManner_tfix.getArg(0).getInt();
      Xobject width_tfix = distManner_tfix.getArg(1);

      if (distMannerValue_decl != distMannerValue_tfix ||
	  (distMannerValue_decl == XMPtemplate.GBLOCK &&
	   (!(width_decl instanceof XobjList) || width_decl.Nargs() != 0))){
	throw new XMPexception("<dist-format> not match that of the distribute directives");
      }
    }

    XobjArgs distMannerArgs = tObject.distributionIsFixed() ? distMannerArgs_decl : distMannerArgs_tfix;

    int tDimIdx = 0;
    int nDim = nObject.getDim();
    int nDimIdx = 0;

    for (i = distMannerArgs; i != null; i = i.nextArgs()) {
      if (tDimIdx == tDim) {
	throw new XMPexception("wrong template dimension indicated, too many");
      }
	
      XobjList distManner = (XobjList)i.getArg();
      int distMannerValue = distManner.getArg(0).getInt();

      Block b = XMPtemplate.setupDistribution(distManner, tObject, tDimIdx, nDimIdx, globalDecl, true, true, pb);
      tFixFuncBody.add(b);

      switch (distMannerValue) {
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
      case XMPtemplate.BLOCK_CYCLIC:
      case XMPtemplate.GBLOCK:
	if (nDimIdx == nDim){
	  throw new XMPexception("the number of <dist-format> (except '*') should be the same with the nodes dimension");
	}

	nDimIdx++;
	break;
      default:
      }

      tDimIdx++;
    }

    // check nodes, template dimension
    if (nDimIdx != nDim){
      throw new XMPexception("the number of <dist-format> (except '*') should be the same with the nodes dimension");
    }

    if (tDimIdx != tDim){
      throw new XMPexception("wrong template dimension indicated, too few");
    }

    tObject.setIsFixed();
    tObject.setDistributionIsFixed();

    Block tFixFuncCallBlock = Bcons.COMPOUND(tFixFuncBody);
    pb.replace(tFixFuncCallBlock);

  }

}
