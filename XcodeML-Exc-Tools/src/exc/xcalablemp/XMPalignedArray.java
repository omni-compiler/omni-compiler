package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.util.Vector;
import java.util.Iterator;

public class XMPalignedArray {
  // defined in xmp_constant.h
  public final static int NOT_ALIGNED	= 200;
  public final static int DUPLICATION	= 201;
  public final static int BLOCK		= 202;
  public final static int CYCLIC	= 203;
  public final static int BLOCK_CYCLIC	= 204;
  public final static int GBLOCK	= 205;

  private String		_name;
  private Xtype			_type;
  private ArrayType		_arrayType;
  private int			_dim;
  private Vector<XMPshadow>	_shadowVector;
  private Vector<Integer>	_alignMannerVector;
  private Vector<Ident>		_accIdVector;
  private Vector<Ident>		_gtolTemp0IdVector;
  private Vector<Integer>	_alignSubscriptIndexVector;
  private Vector<Xobject>	_alignSubscriptExprVector;
  private Vector<Xobject>	_alignNormExprVector;
  private Ident			_arrayId;
  private Ident			_descId;
  private Ident			_addrId;
  private Ident                 _multiAddrId = null;
  private boolean		_hasShadow;
  private boolean		_reallocChecked;
  private boolean		_realloc;
  private XMPtemplate		_alignTemplate;
  private boolean               _isParameter;
  private boolean               _isLocal;
  private boolean               _isPointer;

  private boolean               _isStaticDesc = false;
  private Ident                 _flagId       = null;
  private boolean               _canOptimized = false;
  private boolean               _isStructure  = false;
  private Xobject               _addrObj      = null;

  public void setMultiArrayId(Ident id)
  {
    _multiAddrId = id;
  }

  public Ident getMultiArrayId()
  {
    return _multiAddrId;
  }
  
  public boolean canOptimized()
  {
    return _canOptimized;
  }

  public void setOptimized(boolean flag)
  {
    _canOptimized = flag;
  }
  
  public static int convertDistMannerToAlignManner(int distManner) {
    switch (distManner) {
      case XMPtemplate.DUPLICATION:
        return DUPLICATION;
      case XMPtemplate.BLOCK:
        return BLOCK;
      case XMPtemplate.CYCLIC:
        return CYCLIC;
      case XMPtemplate.BLOCK_CYCLIC:
        return BLOCK_CYCLIC;
      case XMPtemplate.GBLOCK:
	return GBLOCK;
      default:
        XMP.fatal("unknown dist manner");
	return -1;
    }
  }
  
  public XMPalignedArray(String name, Xtype type, ArrayType arrayType,
                         int dim, Vector<Ident> accIdVector,
                         Ident arrayId, Ident descId, Ident addrId,
                         XMPtemplate alignTemplate){
    _name = name;
    _type = type;
    _arrayType = arrayType;
    _dim = dim;
    _shadowVector = new Vector<XMPshadow>(XMP.MAX_DIM);
    _alignMannerVector = new Vector<Integer>(XMP.MAX_DIM);
    _accIdVector = accIdVector;
    _gtolTemp0IdVector = new Vector<Ident>(XMP.MAX_DIM);
    _alignSubscriptIndexVector = new Vector<Integer>(XMP.MAX_DIM);
    _alignSubscriptExprVector = new Vector<Xobject>(XMP.MAX_DIM);
    _alignNormExprVector = new Vector<Xobject>(XMP.MAX_DIM);
    for (int i = 0; i < dim; i++) {
      _shadowVector.add(new XMPshadow(XMPshadow.SHADOW_NONE, null, null));
      _alignMannerVector.add(null);
      _gtolTemp0IdVector.add(null);
      _alignSubscriptIndexVector.add(null);
      _alignSubscriptExprVector.add(null);
      _alignNormExprVector.add(null);
    }
    _arrayId        = arrayId;
    _descId         = descId;
    _addrId         = addrId;
    _hasShadow      = false;
    _reallocChecked = false;
    _alignTemplate  = alignTemplate;
    _isParameter    = false;
    _isLocal        = false;
    _isPointer      = false;
    _addrObj        = addrId.Ref();
  }

  // This constructor is used to define "addrObj" for structure member
  public XMPalignedArray(String name, Xtype type, ArrayType arrayType,
                         int dim, Vector<Ident> accIdVector,
                         Ident arrayId, Ident descId, Ident addrId,
                         XMPtemplate alignTemplate, Xobject addrObj){
    this(name, type, arrayType, dim, accIdVector,
         arrayId, descId, addrId, alignTemplate);
    _addrObj = addrObj;
  }

  public void setStructure(boolean arg)
  {
    _isStructure = arg;
  }
  public boolean isStructure()
  {
    return _isStructure;
  }

  public Xobject getAddrObj(){
    return _addrObj;
  }
  
  public String getName() {
    return _name;
  }

  public Xtype getType() {
    return _type;
  }

  public Xtype getArrayType() {
    return _arrayType;
  }

  public int getDim() {
    return _dim;
  }

  public void setAlignMannerAt(int manner, int index) {
    _alignMannerVector.setElementAt(manner, index);
  }

  public int getAlignMannerAt(int index) {
    return _alignMannerVector.get(index).intValue();
  }

  public String getAlignMannerStringAt(int index) throws XMPexception {
    switch (getAlignMannerAt(index)) {
      case NOT_ALIGNED:
        return new String("NOT_ALIGNED");
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
        throw new XMPexception("unknown align manner");
    }
  }

  public Vector<Ident> getAccIdVector() {
    return _accIdVector;
  }

  public Ident getAccIdAt(int index) {
    return _accIdVector.get(index);
  }

  // temp0 is
  // block distribution:	parallel/serial lower	| _XMP_gtol_lower_<array_name>_<array_dim>
  // cyclic distribution:	nodes size		| _XMP_gtol_cycle_<array_name>_<array_dim>
  public void setGtolTemp0IdAt(Ident temp0Id, int index) {
    _gtolTemp0IdVector.setElementAt(temp0Id, index);
  }

  public Ident getGtolTemp0IdAt(int index) {
    return _gtolTemp0IdVector.get(index);
  }

  public void setAlignSubscriptIndexAt(int alignSubscriptIndex, int alignSourceIndex) {
    _alignSubscriptIndexVector.setElementAt(alignSubscriptIndex, alignSourceIndex);
  }

  public Integer getAlignSubscriptIndexAt(int alignSourceIndex) {
    return _alignSubscriptIndexVector.get(alignSourceIndex);
  }

  public void setAlignSubscriptExprAt(Xobject alignSubscriptExpr, int alignSourceIndex) {
    _alignSubscriptExprVector.setElementAt(alignSubscriptExpr, alignSourceIndex);
  }

  public Xobject getAlignSubscriptExprAt(int alignSourceIndex) {
    return _alignSubscriptExprVector.get(alignSourceIndex);
  }

  public void setAlignNormExprAt(Xobject alignNormExpr, int index) {
    _alignNormExprVector.setElementAt(alignNormExpr, index);
  }

  public Xobject getAlignNormExprAt(int index) {
    return _alignNormExprVector.get(index);
  }

  public Ident getArrayId() {
    return _arrayId;
  }

  public Ident getDescId() {
    return _descId;
  }

  public Ident getAddrId() {
    return _addrId;
  }

  public Xobject getAddrIdVoidRef() {
    return Xcons.Cast(Xtype.voidPtrType, _addrId.Ref());
  }

  public Xobject getAddrIdVoidAddr() {
    return Xcons.Cast(Xtype.Pointer(Xtype.voidPtrType), _addrId.getAddr());
  }

  public void setHasShadow() {
    _hasShadow = true;
  }

  public boolean hasShadow() {
    return _hasShadow;
  }

  public void setIsParameter() {
    _isParameter = true;
  }

  public boolean isParameter() {
    return _isParameter;
  }

  public void setIsLocal() {
    _isLocal = true;
  }

  public boolean isLocal() {
    return _isLocal;
  }

  public void setIsPointer() {
    _isPointer = true;
  }

  public boolean isPointer() {
    return _isPointer;
  }

  public void setShadowAt(XMPshadow shadow, int index) {
    _shadowVector.setElementAt(shadow, index);
  }

  public XMPshadow getShadowAt(int index) {
    return _shadowVector.get(index);
  }

  public XMPtemplate getAlignTemplate() {
    return _alignTemplate;
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

  public boolean checkRealloc() throws XMPexception {
    if (_reallocChecked) return _realloc;

    if (_hasShadow) {
      for (int i = 0; i < _dim; i++) {
        switch (getAlignMannerAt(i)) {
          case NOT_ALIGNED:
          case DUPLICATION:
            break;
          case BLOCK:
          case CYCLIC:
          case BLOCK_CYCLIC:
	  case GBLOCK:
            {
              XMPshadow shadow = getShadowAt(i);
              switch (shadow.getType()) {
                case XMPshadow.SHADOW_FULL:
                  break;
                case XMPshadow.SHADOW_NONE:
                case XMPshadow.SHADOW_NORMAL:
                  {
                    _reallocChecked = true;
                    _realloc = true;
                    return _realloc;
                  }
                default:
                  throw new XMPexception("unknown shadow type");
              }
            } break;
          default:
            throw new XMPexception("unknown align manner");
        }
      }

      _reallocChecked = true;
      _realloc = false;
      return _realloc;
    }
    else {
      for (int i = 0; i < _dim; i++) {
        switch (getAlignMannerAt(i)) {
          case NOT_ALIGNED:
          case DUPLICATION:
            break;
          case BLOCK:
          case CYCLIC:
          case BLOCK_CYCLIC:
	  case GBLOCK:
            {
              _reallocChecked = true;
              _realloc = true;
              return _realloc;
            }
          default:
            throw new XMPexception("unknown align manner");
        }
      }

      _reallocChecked = true;
      _realloc = false;
      return _realloc;
    }
  }

  public boolean realloc() throws XMPexception {
    if (_reallocChecked) return _realloc;
    else                 return checkRealloc();
  }

  public void normArraySize(int index, Xobject normExpr) {
    ArrayType type = this._arrayType;
    for (int i = 0; i < index; i++) {
      type = (ArrayType)type.getRef();
    }

    // FIXME case (size == 0) ???
    long size = type.getArraySize();
    if (size == -1) {
      type.setArraySizeExpr(Xcons.binaryOp(Xcode.PLUS_EXPR, type.getArraySizeExpr(), normExpr));
    } else {
      type.setArraySize(-1);
      type.setArraySizeExpr(Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.LongLongConstant(0, size), normExpr));
    }
  }

  private static Boolean is_SameSizeTemplateArray(XMPalignedArray alignedArray) {
    XMPtemplate t   = alignedArray.getAlignTemplate();
    int arrayDim    = alignedArray.getDim();
    Xtype arrayType = alignedArray.getArrayType();
    for (int i=0; i<arrayDim; i++, arrayType=arrayType.getRef()){
      if(arrayType.getArraySize() == 0)  // Use xmp_malloc
        return false;

      switch (alignedArray.getAlignMannerAt(i)){
      case XMPtemplate.GBLOCK:
        return false;
      case XMPalignedArray.BLOCK:
      case XMPalignedArray.CYCLIC:
      case XMPalignedArray.BLOCK_CYCLIC:
        Xobject x = arrayType.getArraySizeExpr();
        if(x.isConstant() == false) return false;
        if(x.getLongHigh() != 0)    return false; // fix me
        int index = alignedArray.getAlignSubscriptIndexAt(i);
        int template_size = XMPutil.foldIntConstant(t.getSizeAt(index)).getInt();
        if((int)x.getLongLow() != template_size) return false;
      }
    }

    return true;
  }

  // Is size of template % size of node == 0 && size of template and size of array?
  private static Boolean is_divisible_size(XMPalignedArray alignedArray) {
    XMPtemplate t = alignedArray.getAlignTemplate();
    XMPnodes    n = t.getOntoNodes();
    if(! XMPutil.is_AllConstant(t))              return false;
    if(! XMPutil.is_AllConstant(n))              return false;
    if(! is_SameSizeTemplateArray(alignedArray)) return false;

    // Number of dimensions of template must be larger than that of node.
    for(int i=0;i<t.getDim();i++){
      int manner = t.getDistMannerAt(i);

      switch (manner){
      case XMPtemplate.GBLOCK:
        return false;
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
      case XMPtemplate.BLOCK_CYCLIC:
        int template_size = XMPutil.foldIntConstant(t.getSizeAt(i)).getInt();
        int blocksize     = (manner == XMPtemplate.BLOCK_CYCLIC)? XMPutil.foldIntConstant(t.getWidthAt(i)).getInt() : 1;
        int node_rank     = t.getOntoNodesIndexAt(i).getInt();
        int node_size     = XMPutil.foldIntConstant(n.getSizeAt(node_rank)).getInt();
        if(template_size%(node_size*blocksize) != 0)
          return false;
        break;
      }
    }
    return true;
  }

  public static void createAlignFunctionCalls(XMPalignedArray alignedArray, XMPglobalDecl globalDecl,
					      XobjList alignSourceList, XobjList alignSubscriptVarList,
					      XobjList alignSubscriptExprList, XMPtemplate templateObj,
					      PragmaBlock pb, Block parentBlock, Ident arrayId, int arrayDim,
					      String arrayName, Ident arrayDescId, boolean isLocalPragma,
					      boolean isPointer, boolean isParameter, boolean isStaticDesc)
  {
    int alignSourceIndex = 0;
    for (XobjArgs i = alignSourceList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSource = i.getArg().getString();

      if (alignSource.equals(XMP.ASTERISK)) {
	if (!isPointer)
	  declNotAlignFunc(alignedArray, alignSourceIndex, globalDecl, isLocalPragma, pb);
	else
	  declAlignFunc_pointer(alignedArray, alignSourceIndex, null, -1,
				Xcons.IntConstant(0), globalDecl, isLocalPragma, pb);
      }
      else if (alignSource.equals(XMP.COLON)) {
	if (!XMPutil.hasElmt(alignSubscriptVarList, XMP.COLON))
	  XMP.fatal("cannot find ':' in <align-subscript> list");
	
	int alignSubscriptIndex = XMPutil.getLastIndex(alignSubscriptVarList, XMP.COLON);
	alignSubscriptVarList.setArg(alignSubscriptIndex, null);

	if (!isPointer)
	  declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
			alignSubscriptExprList.getArg(alignSubscriptIndex), globalDecl, isLocalPragma, pb);
	else
	  declAlignFunc_pointer(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
				alignSubscriptExprList.getArg(alignSubscriptIndex), globalDecl, isLocalPragma, pb);
      }
      else {
	if (XMPutil.countElmts(alignSourceList, alignSource) != 1)
	  XMP.fatal("multiple '" + alignSource + "' indicated in <align-source> list");

	if (XMPutil.hasElmt(alignSubscriptVarList, alignSource)) {
	  if (XMPutil.countElmts(alignSubscriptVarList, alignSource) != 1)
	    XMP.fatal("multiple '" + alignSource + "' indicated in <align-subscript> list");

	  int alignSubscriptIndex = XMPutil.getFirstIndex(alignSubscriptVarList, alignSource);
	  if (!isPointer)
	    declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
			  alignSubscriptExprList.getArg(alignSubscriptIndex), globalDecl, isLocalPragma, pb);
	  else
	    declAlignFunc_pointer(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
				  alignSubscriptExprList.getArg(alignSubscriptIndex), globalDecl, isLocalPragma, pb);
	}
	else {
	  XMP.fatal("cannot find '" + alignSource + "' in <align-subscript> list");
	}
      }
      alignSourceIndex++;
    }
    
    if (isPointer) return;

    // check alignSubscriptVarList
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();
      
      if (alignSubscript.equals(XMP.ASTERISK) || alignSubscript.equals(XMP.COLON))
	break;
      
      if (XMPutil.hasElmt(alignSourceList, alignSubscript)) {
	if (XMPutil.countElmts(alignSourceList, alignSubscript) != 1)
	  XMP.fatal("no/multiple '" + alignSubscript + "' indicated in <align-source> list");
      }
      else
	XMP.fatal("cannot find '" + alignSubscript + "' in <align-source> list");
    }
    
    // init array communicator
    XobjList initArrayCommFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();
      if (alignSubscript.equals(XMP.ASTERISK))
	initArrayCommFuncArgs.add(Xcons.IntConstant(1));
      else
	initArrayCommFuncArgs.add(Xcons.IntConstant(0));
    }

    if (isLocalPragma) {
      if (isStaticDesc){
	XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_init_array_comm", initArrayCommFuncArgs, globalDecl, parentBlock,
						    alignedArray.getFlagId(), false);
	XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_init_array_nodes", Xcons.List(alignedArray.getDescId().Ref()), globalDecl,
						    parentBlock, alignedArray.getFlagId(), false);
      }
      else {
	XMPlocalDecl.addConstructorCall2("_XMP_init_array_comm", initArrayCommFuncArgs, globalDecl, parentBlock);
	XMPlocalDecl.addConstructorCall2("_XMP_init_array_nodes", Xcons.List(alignedArray.getDescId().Ref()), globalDecl, parentBlock);
      }

      if (isParameter){
	// init array address
	XobjList initArrayAddrFuncArgs = Xcons.List(alignedArray.getAddrIdVoidAddr(),
						    arrayId.Ref(), arrayDescId.Ref());

	for (int i = arrayDim - 1; i >= 0; i--)
	  initArrayAddrFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
					       alignedArray.getAccIdAt(i).getAddr()));

	XMPlocalDecl.addAllocCall2("_XMP_init_array_addr", initArrayAddrFuncArgs, globalDecl, parentBlock);
	XobjList bodyList = (XobjList)parentBlock.getProp("XCALABLEMP_PROP_LOCAL_ALLOC");
	if (isStaticDesc)
	  bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, Xcons.Set(alignedArray.getFlagId().Ref(), Xcons.IntConstant(1))));
      }
      else {
	Xobject isCoarray = (arrayId.getStorageClass() == StorageClass.EXTDEF) ? Xcons.IntConstant(1) : Xcons.IntConstant(0);
	XobjList allocFuncArgs = Xcons.List(alignedArray.getAddrIdVoidAddr(), alignedArray.getDescId().Ref(), isCoarray);
	for (int i = alignedArray.getDim() - 1; i >= 0; i--)
	  allocFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
				       alignedArray.getAccIdAt(i).getAddr()));

	XMPlocalDecl.addAllocCall2("_XMP_alloc_array", allocFuncArgs, globalDecl, parentBlock);
	XobjList bodyList = (XobjList)parentBlock.getProp("XCALABLEMP_PROP_LOCAL_ALLOC");
	if (isStaticDesc)
	  bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, Xcons.Set(alignedArray.getFlagId().Ref(), Xcons.IntConstant(1))));
	
	XMPlocalDecl.insertDestructorCall2("_XMP_dealloc_array", Xcons.List(alignedArray.getDescId().Ref()),
					   globalDecl, parentBlock);
      }
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_init_array_comm", initArrayCommFuncArgs);
      globalDecl.addGlobalInitFuncCall("_XMP_init_array_nodes", Xcons.List(alignedArray.getDescId().Ref()));
    }

    if (isLocalPragma && !isParameter)
      XMPlocalDecl.removeLocalIdent(pb, arrayName);

    if(arrayDim > 1 && is_divisible_size(alignedArray))
      alignedArray.setOptimized(true);
    else
      alignedArray.setOptimized(false);
  }

  public static void translateAlign(XobjList alignDecl, XMPglobalDecl globalDecl,
				    boolean isLocalPragma, PragmaBlock pb,
				    Xobject structVar) throws XMPexception
  {
    String arrayName = alignDecl.getArg(0).getString();
    XMPsymbolTable localXMPsymbolTable = null;
    Ident arrayId        = null;
    Block parentBlock    = null;
    Boolean isPointer    = false;
    boolean isStaticDesc = false;
    Boolean isParameter  = isLocalPragma;
    Boolean isStructure  = (structVar != null);

    if(isStructure){
      if(isLocalPragma)
	throw new XMPexception("structure member can be decleared in only global scope");
      
      String structName = structVar.getString();
      Ident  structId   = globalDecl.findIdent(structName);
      if (structId == null) {
        throw new XMPexception("structure '" + structName + "' is not declared");
      }
      arrayId = structId.Type().getMemberList().getIdent(arrayName);
      arrayId.saveOrigId();
    }
    else{
      if (isLocalPragma) {
	arrayId      = XMPlocalDecl.findLocalIdent(pb, arrayName);
	parentBlock  = pb.getParentBlock();
	isStaticDesc = XMPlocalDecl.declXMPsymbolTable2(parentBlock).isStaticDesc(arrayName);
	
	if (arrayId != null)
	  isParameter = (arrayId.getStorageClass() == StorageClass.PARAM);
	
	if (isParameter){
	  localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
	}
	else {
	  localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable2(parentBlock);
	}
	isStaticDesc = XMPlocalDecl.declXMPsymbolTable2(parentBlock).isStaticDesc(arrayName);
      }
      else {
	arrayId = globalDecl.findVarIdent(arrayName);
      }
    }
    
    if (arrayId == null)
      throw new XMPexception("array '" + arrayName + "' is not declared");

    if (isStructure)
      arrayId.setMemberAligned(true);
    
    // get array information
    XMPalignedArray alignedArray = null;
    if (!isStructure){ // At this point, a structure variable is not decleared.
      if (isLocalPragma)
	alignedArray = localXMPsymbolTable.getXMPalignedArray(arrayName);
      else
	alignedArray = globalDecl.getXMPalignedArray(arrayName);
    
      if (alignedArray != null)
	throw new XMPexception("array '" + arrayName + "' is already aligned");
    }
    
    Xtype arrayType = arrayId.Type();
    if (arrayType.getKind() != Xtype.ARRAY){
      if (arrayType.getKind() == Xtype.POINTER){
	isPointer = true;
	arrayType = new ArrayType(arrayType.getRef(), 0l);
      }
      else {
	throw new XMPexception(arrayName + " is neither an array nor pointer");
      }
    }
    
    Xtype arrayElmtType      = arrayType.getArrayElementType();
    Xobject arrayElmtTypeRef = null;
    if (arrayElmtType.getKind() == Xtype.BASIC)
      arrayElmtTypeRef = XMP.createBasicTypeConstantObj(arrayElmtType);
    else
      arrayElmtTypeRef = Xcons.IntConstant(XMP.NONBASIC_TYPE);
    
    // check coarray table
    if (globalDecl.getXMPcoarray(arrayName, pb) != null && !isStructure)
      throw new XMPexception("array '" + arrayName + "' is declared as a coarray, cannot be aligned");
    
    // declare array address pointer, array descriptor
    Ident arrayAddrId = arrayId;
    Ident arrayDescId = null;
    if (isStructure) {
      Xtype newArrayType = Xtype.Pointer(arrayElmtType);
      arrayAddrId.setType(newArrayType);
      arrayAddrId.setName(XMP.ADDR_PREFIX_ + arrayName);
      arrayAddrId.setValue(Xcons.Symbol(Xcode.VAR_ADDR, Xtype.Pointer(newArrayType), arrayId.getSym()));
    }
    else{
      if (isLocalPragma) {
	if (!isPointer) {
	  arrayAddrId = XMPlocalDecl.addObjectId2(XMP.ADDR_PREFIX_ + arrayName,
						  Xtype.Pointer(arrayElmtType), parentBlock);
	}
	else {
	  Xtype newArrayAddrType = Xtype.Pointer(arrayElmtType);
	  arrayAddrId.setType(newArrayAddrType);
	  arrayAddrId.setValue(Xcons.Symbol(Xcode.VAR_ADDR, Xtype.Pointer(newArrayAddrType), arrayAddrId.getSym(), VarScope.LOCAL));
	}
	arrayDescId = XMPlocalDecl.addObjectId2(XMP.DESC_PREFIX_ + arrayName, parentBlock);
      }
      else {
	if (!isPointer){
	  if (arrayId.getStorageClass() == StorageClass.EXTERN) {
	    arrayAddrId = globalDecl.declExternIdent(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType));
	  }
	  else if (arrayId.getStorageClass() == StorageClass.STATIC) {
	    arrayAddrId = globalDecl.declStaticIdent(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType));
	  }
	  else if (arrayId.getStorageClass() == StorageClass.EXTDEF) {
	    arrayAddrId = globalDecl.declGlobalIdent(XMP.ADDR_PREFIX_ + arrayName, Xtype.Pointer(arrayElmtType));
	  }
	  else {
	    throw new XMPexception("cannot align array '" + arrayName + "', wrong storage class");
	  }
	}
	else {
	  Xtype newArrayAddrType = Xtype.Pointer(arrayElmtType);
	  arrayAddrId.setType(newArrayAddrType);
	  arrayAddrId.setValue(Xcons.Symbol(Xcode.VAR_ADDR, Xtype.Pointer(newArrayAddrType), arrayAddrId.getSym(), VarScope.GLOBAL));
	}
	arrayDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + arrayName, Xtype.voidPtrType);
      }
    }
    
    if (isStaticDesc) arrayDescId.setStorageClass(StorageClass.STATIC);
    
    // get template information
    String templateName     = alignDecl.getArg(2).getString();
    XMPtemplate templateObj = globalDecl.getXMPtemplate(templateName, pb);
    
    if (templateObj == null)
      throw new XMPexception("template '" + templateName + "' is not declared");
    else if (! templateObj.isDistributed())
      throw new XMPexception("template '" + templateName + "' is not distributed");
    
    int templateDim = templateObj.getDim();
    int arrayDim = arrayType.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM)
      throw new XMPexception("array dimension should be less than " + (XMP.MAX_DIM + 1));

    XobjList initArrayDescFuncArgs;
    if(isStructure) // At this point, arrayDescId is not created, so it is inserted in XMPrewriteExpr.rewriteVarDecl().
      initArrayDescFuncArgs = Xcons.List(templateObj.getDescId().Ref(), Xcons.IntConstant(arrayDim),
					arrayElmtTypeRef, Xcons.SizeOf(arrayElmtType));
    else
      initArrayDescFuncArgs = Xcons.List(arrayDescId.getAddr(),templateObj.getDescId().Ref(),
					Xcons.IntConstant(arrayDim), arrayElmtTypeRef,
					Xcons.SizeOf(arrayElmtType));

    if(isStructure){
      arrayAddrId.setProp(XMP.DESC_FUNC_ARGS, initArrayDescFuncArgs);
      arrayAddrId.setProp(XMP.TEMPLATE, templateObj);
    }
    else{
      Vector<Ident> accIdVector = new Vector<Ident>(arrayDim);
      for (int i=0;i<arrayDim;i++) {
	Ident accId = null;
	String accName = XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i;
	if (isLocalPragma) {
	  accId = XMPlocalDecl.addObjectId2(accName, Xtype.unsignedlonglongType, parentBlock);
	  if (isStaticDesc) accId.setStorageClass(StorageClass.STATIC);
	}
	else {
	  accId = globalDecl.declStaticIdent(accName, Xtype.unsignedlonglongType);
	}
	accIdVector.add(accId);
      }

      alignedArray = new XMPalignedArray(arrayName, arrayElmtType, (ArrayType)arrayType, arrayDim,
					 accIdVector, arrayId, arrayDescId, arrayAddrId, templateObj);

      if (isLocalPragma){
	alignedArray.setIsLocal();
	alignedArray.setIsStaticDesc(isStaticDesc);
      }
      if (isParameter) alignedArray.setIsParameter();
      if (isPointer)   alignedArray.setIsPointer();
      if (isStaticDesc && isPointer)
	throw new XMPexception("a pointer cannot have the static_desc attribute.");
    
      if (isLocalPragma) {
	if (isStaticDesc){
	  Ident id = parentBlock.getBody().declLocalIdent(XMP.STATIC_DESC_PREFIX_ + arrayName, Xtype.intType,
							  StorageClass.STATIC, Xcons.IntConstant(0));
	  alignedArray.setFlagId(id);
	  XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_init_array_desc", initArrayDescFuncArgs, globalDecl, parentBlock,
						      id, false);
	}
	else {
	  XMPlocalDecl.addConstructorCall2("_XMP_init_array_desc", initArrayDescFuncArgs, globalDecl, parentBlock);
	}
	
	if (!isStaticDesc)
	  XMPlocalDecl.insertDestructorCall2("_XMP_finalize_array_desc", Xcons.List(arrayDescId.Ref()), globalDecl, parentBlock);
	localXMPsymbolTable.putXMPalignedArray(alignedArray);
      }
      else {
	globalDecl.addGlobalInitFuncCall("_XMP_init_array_desc", initArrayDescFuncArgs);
	globalDecl.putXMPalignedArray(alignedArray);
      }
    }

    // check <align-source> list, <align-subscrip> list
    XobjList alignSourceList        = (XobjList)alignDecl.getArg(1);
    XobjList alignSubscriptList     = (XobjList)alignDecl.getArg(3);
    XobjList alignSubscriptVarList  = (XobjList)alignSubscriptList.left();
    XobjList alignSubscriptExprList = (XobjList)alignSubscriptList.right();

    if(arrayType.getRef().getKind() == Xtype.POINTER)   // check <align-source> list
      throw new XMPexception("Pointer to pointer \"" + arrayName + "\" can not be used.");
    else if (XMPutil.countElmts(alignSourceList) != arrayDim) 
      throw new XMPexception("the number of <align-source>s is not the same with array dimension");
    else if (XMPutil.countElmts(alignSourceList, XMP.ASTERISK) == arrayDim)
      throw new XMPexception("array " + arrayName + " is not aligned on any dimension");
    else if (XMPutil.countElmts(alignSubscriptVarList) != templateDim)     // check <align-subscript> list
      throw new XMPexception("the number of <align-subscript>s is not the same with template dimension");
    else if (XMPutil.countElmts(alignSourceList, XMP.COLON) !=
	     XMPutil.countElmts(alignSubscriptVarList, XMP.COLON))  // check ':' source/subscript
      throw new XMPexception("the number of ':' in <align-source> list is not the same with <align-subscript> list");

    if (!isPointer){
      // add array size to args: do this after declAlignFunc()
      for (int i = 0; i < arrayDim; i++, arrayType = arrayType.getRef()) {
	long dimSize = arrayType.getArraySize();
	if (dimSize == 0)
	  throw new XMPexception("array size should be declared statically");
	else if (dimSize == -1)
	  initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, arrayType.getArraySizeExpr()));
	else
	  initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));
      }
    }

    if(isStructure){
      arrayId.setProp(XMP.ALIGN_SOURCE_LIST,         alignSourceList);
      arrayId.setProp(XMP.ALIGN_SUBSCRIPT_VAR_LIST,  alignSubscriptVarList);
      arrayId.setProp(XMP.ALIGN_SUBSCRIPT_EXPR_LIST, alignSubscriptExprList);
      arrayId.setProp(XMP.PRAGMA_BLOCK, pb);
      arrayId.setProp(XMP.PARENT_BLOCK, parentBlock);
    }
    else{
      createAlignFunctionCalls(alignedArray, globalDecl, alignSourceList, alignSubscriptVarList,
			       alignSubscriptExprList, templateObj, pb, parentBlock, arrayId, arrayDim,
			       arrayName, arrayDescId, isLocalPragma, isPointer, isParameter, isStaticDesc);
    }
  }

  private static void declNotAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                                       XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) {
    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex));

    alignedArray.setAlignMannerAt(XMPalignedArray.NOT_ALIGNED, alignSourceIndex);

    if (isLocalPragma) {
      Block parentBlock = pb.getParentBlock();
      if (alignedArray.isStaticDesc()){
	XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_align_array_NOT_ALIGNED", alignFuncArgs, globalDecl, parentBlock,
						    alignedArray.getFlagId(), false);
      }
      else {
	XMPlocalDecl.addConstructorCall2("_XMP_align_array_NOT_ALIGNED", alignFuncArgs, globalDecl, parentBlock);
      }
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_align_array_NOT_ALIGNED", alignFuncArgs);
    }
  }

  private static Xobject normArray(XMPalignedArray alignedArray, int alignSourceIndex,
                                   XMPtemplate templateObj, int alignSubscriptIndex,
                                   Xobject alignSubscriptExpr,
                                   XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) {
    Xobject templateLower = templateObj.getLowerAt(alignSubscriptIndex);
    Xobject alignNormExpr = Xcons.binaryOp(Xcode.MINUS_EXPR,
                                           alignSubscriptExpr, templateLower);
    alignedArray.setAlignNormExprAt(alignNormExpr, alignSourceIndex);

    // normalize 1. array size on src code: += normExpr
    alignedArray.normArraySize(alignSourceIndex, alignNormExpr);

    // normalize 2. alignSubscriptExpr: templateLower
    return templateLower;
  }

  private static void declAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex, XMPtemplate templateObj,
				    int alignSubscriptIndex, Xobject alignSubscriptExpr,
                                    XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) {

    Block parentBlock = null;
    if (isLocalPragma) parentBlock = pb.getParentBlock();

    // normalize array
    alignSubscriptExpr = normArray(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
                                   alignSubscriptExpr, globalDecl, isLocalPragma, pb);

    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex),
                                        Xcons.IntConstant(alignSubscriptIndex));

    alignFuncArgs.add(alignSubscriptExpr);

    int distManner = templateObj.getDistMannerAt(alignSubscriptIndex);
    alignedArray.setAlignMannerAt(convertDistMannerToAlignManner(distManner), alignSourceIndex);

    alignedArray.setAlignSubscriptIndexAt(alignSubscriptIndex, alignSourceIndex);
    alignedArray.setAlignSubscriptExprAt(alignSubscriptExpr, alignSourceIndex);

    switch (distManner) {
      case XMPtemplate.DUPLICATION: // FIXME how implement???
        break;
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
      case XMPtemplate.BLOCK_CYCLIC:
      case XMPtemplate.GBLOCK:
        {
          Ident gtolTemp0Id = null;
          if (isLocalPragma) {
            gtolTemp0Id = XMPlocalDecl.addObjectId2(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
						    Xtype.intType, parentBlock);
	    if (alignedArray.isStaticDesc()) gtolTemp0Id.setStorageClass(StorageClass.STATIC);
          }
          else {
            gtolTemp0Id = globalDecl.declStaticIdent(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
                                                     Xtype.intType);
          }

          alignedArray.setGtolTemp0IdAt(gtolTemp0Id, alignSourceIndex);
          alignFuncArgs.add(gtolTemp0Id.getAddr());

          break;
        }
      default:
        XMP.fatal("unknown distribute manner");
    }

    if (isLocalPragma) {
      if (alignedArray.isStaticDesc()){
	XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
						    alignFuncArgs, globalDecl, parentBlock, alignedArray.getFlagId(), false);
      }
      else {
	XMPlocalDecl.addConstructorCall2("_XMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
					 alignFuncArgs, globalDecl, parentBlock);
      }
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_align_array_" + templateObj.getDistMannerStringAt(alignSubscriptIndex),
                                       alignFuncArgs);
    }
  }

  private static void declAlignFunc_pointer(XMPalignedArray alignedArray, int alignSourceIndex, XMPtemplate templateObj,
					    int alignSubscriptIndex, Xobject alignSubscriptExpr,
					    XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) {

    Block parentBlock = null;
    if (isLocalPragma) parentBlock = pb.getParentBlock();

    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex),
                                        Xcons.IntConstant(alignSubscriptIndex));

    alignFuncArgs.add(alignSubscriptExpr);

    if (templateObj != null){
      int distManner = templateObj.getDistMannerAt(alignSubscriptIndex);
      alignedArray.setAlignMannerAt(XMPalignedArray.convertDistMannerToAlignManner(distManner), alignSourceIndex);

      alignedArray.setAlignSubscriptIndexAt(alignSubscriptIndex, alignSourceIndex);
      alignedArray.setAlignSubscriptExprAt(alignSubscriptExpr, alignSourceIndex);
    }
    else {
      alignedArray.setAlignMannerAt(XMPalignedArray.NOT_ALIGNED, alignSourceIndex);
    }

    Ident gtolTemp0Id = null;
    if (isLocalPragma) {
      gtolTemp0Id = XMPlocalDecl.addObjectId2(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
					      Xtype.intType, parentBlock);
      if (alignedArray.isStaticDesc()) gtolTemp0Id.setStorageClass(StorageClass.STATIC);
    }
    else {
      gtolTemp0Id = globalDecl.declStaticIdent(XMP.GTOL_PREFIX_ + "temp0_" + alignedArray.getName() + "_" + alignSourceIndex,
					       Xtype.intType);
    }

    alignedArray.setGtolTemp0IdAt(gtolTemp0Id, alignSourceIndex);
    alignFuncArgs.add(gtolTemp0Id.getAddr());

    // assumed that dynamic arrays are one-dimentional.
    alignFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
				 alignedArray.getAccIdAt(alignSourceIndex).getAddr()));

    if (isLocalPragma) {
      if (alignedArray.isStaticDesc()){
	XMPlocalDecl.addConstructorCall2_staticDesc("_XMP_align_array_noalloc", alignFuncArgs, globalDecl, parentBlock,
						    alignedArray.getFlagId(), false);
      }
      else {
	XMPlocalDecl.addConstructorCall2("_XMP_align_array_noalloc", alignFuncArgs, globalDecl, parentBlock);
      }
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_align_array_noalloc", alignFuncArgs);
    }
  }

  // FIXME implement
  public static void translateLocalAlias(XobjList localAliasDecl, XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // start translation
    XMPsymbolTable localXMPsymbolTable = null;
    if (isLocalPragma) {
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    }

    // check local array
    String localArrayName = localAliasDecl.getArg(0).getString();
    if (XMPutil.findXMPalignedArray(localArrayName, globalDecl, localXMPsymbolTable) != null) {
      throw new XMPexception("array '" + localArrayName + "' is declared as an aligned array");
    }

    // check pointer

    // check global array
    String globalArrayName = localAliasDecl.getArg(1).getString();
    XMPalignedArray alignedArray = XMPutil.findXMPalignedArray(globalArrayName, globalDecl, localXMPsymbolTable);
    if (alignedArray == null) {
      throw new XMPexception("the aligned array '" + globalArrayName + "' is not found");
    }

    // create runtime func call
  }
}
