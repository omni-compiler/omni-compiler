/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

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
  private boolean		_hasShadow;
  private boolean		_reallocChecked;
  private boolean		_realloc;
  private XMPtemplate		_alignTemplate;
  private boolean               _isParameter;
  private boolean               _isLocal;
  private boolean               _isPointer;

  private boolean               _isStaticDesc = false;
  private Ident                 _flagId = null;

  public static int convertDistMannerToAlignManner(int distManner) throws XMPexception {
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
        throw new XMPexception("unknown dist manner");
    }
  }

  public XMPalignedArray(String name, Xtype type, ArrayType arrayType,
                         int dim, Vector<Ident> accIdVector,
                         Ident arrayId, Ident descId, Ident addrId,
                         XMPtemplate alignTemplate) {
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
    _arrayId = arrayId;
    _descId = descId;
    _addrId = addrId;
    _hasShadow = false;
    _reallocChecked = false;
    _alignTemplate = alignTemplate;
    _isParameter = false;
    _isLocal = false;
    _isPointer = false;
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
    _alignMannerVector.setElementAt(new Integer(manner), index);
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
    _alignSubscriptIndexVector.setElementAt(new Integer(alignSubscriptIndex), alignSourceIndex);
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

  public void normArraySize(int index, Xobject normExpr) throws XMPexception {
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

  public static void translateAlign(XobjList alignDecl, XMPglobalDecl globalDecl,
                                    boolean isLocalPragma, PragmaBlock pb) throws XMPexception {

    String arrayName = alignDecl.getArg(0).getString();
    Ident arrayId = null;

    XMPsymbolTable localXMPsymbolTable = null;
    Block parentBlock = null;

    Boolean isParameter = isLocalPragma;
    Boolean isPointer = false;
    boolean isStaticDesc = false;

    if (isLocalPragma) {
      arrayId = XMPlocalDecl.findLocalIdent(pb, arrayName);
      if (arrayId != null) isParameter = (arrayId.getStorageClass() == StorageClass.PARAM);

      parentBlock = pb.getParentBlock();

      if (isParameter){
	localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
      }
      else {
	localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable2(parentBlock);
      }
      //isStaticDesc = localXMPsymbolTable.isStaticDesc(arrayName);
      isStaticDesc = XMPlocalDecl.declXMPsymbolTable2(parentBlock).isStaticDesc(arrayName);
    }
    else {
      arrayId = globalDecl.findVarIdent(arrayName);
    }

    if (arrayId == null) {
      throw new XMPexception("array '" + arrayName + "' is not declared");
    }

    // get array information
    XMPalignedArray alignedArray = null;
    if (isLocalPragma) {
      alignedArray = localXMPsymbolTable.getXMPalignedArray(arrayName);
    }
    else {
      alignedArray = globalDecl.getXMPalignedArray(arrayName);
    }

    if (alignedArray != null) {
      throw new XMPexception("array '" + arrayName + "' is already aligned");
    }

    Xtype arrayType = arrayId.Type();
    if (arrayType.getKind() == Xtype.ARRAY){
      ;
    }
    else if (arrayType.getKind() == Xtype.POINTER){
      isPointer = true;
      arrayType = new ArrayType(arrayType.getRef(), 0l);
    }
    else {
      throw new XMPexception(arrayName + " is neither an array nor pointer");
    }

    Xtype arrayElmtType = arrayType.getArrayElementType();
    Xobject arrayElmtTypeRef = null;
    if (arrayElmtType.getKind() == Xtype.BASIC) {
      arrayElmtTypeRef = XMP.createBasicTypeConstantObj(arrayElmtType);
    }
    else {
      arrayElmtTypeRef = Xcons.IntConstant(XMP.NONBASIC_TYPE);
    }

    // check coarray table
    //if (globalDecl.getXMPcoarray(arrayName, localXMPsymbolTable) != null) {
    if (globalDecl.getXMPcoarray(arrayName, pb) != null) {
      throw new XMPexception("array '" + arrayName + "' is declared as a coarray, cannot be aligned");
    }

    // get template information
    String templateName = alignDecl.getArg(2).getString();
    XMPtemplate templateObj = globalDecl.getXMPtemplate(templateName, pb);
    // if (isLocalPragma)
    //   for (Block b = parentBlock; b != null; b = b.getParentBlock()){
    // 	templateObj = globalDecl.getXMPtemplate(templateName, XMPlocalDecl.getXMPsymbolTable2(b));
    // 	if (templateObj != null) break;
    //   }
    // else 
    //   templateObj = globalDecl.getXMPtemplate(templateName, localXMPsymbolTable);

    if (templateObj == null) {
      throw new XMPexception("template '" + templateName + "' is not declared");
    }

    if (!templateObj.isFixed() && !isPointer) {
      throw new XMPexception("An array cannot aligned with a non-fixed template '" + templateName +"'");
    }

    if (!(templateObj.isDistributed())) {
      throw new XMPexception("template '" + templateName + "' is not distributed");
    }

    int templateDim = templateObj.getDim();

    // declare array address pointer, array descriptor
    Ident arrayAddrId = arrayId;
    Ident arrayDescId = null;
    if (isLocalPragma) {
      if (!isPointer) arrayAddrId = XMPlocalDecl.addObjectId2(XMP.ADDR_PREFIX_ + arrayName,
							      Xtype.Pointer(arrayElmtType), parentBlock);
      else arrayAddrId.setType(Xtype.Pointer(arrayElmtType));
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
      else arrayAddrId.setType(Xtype.Pointer(arrayElmtType));
      arrayDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + arrayName, Xtype.voidPtrType);
    }

    if (isStaticDesc) arrayDescId.setStorageClass(StorageClass.STATIC);

    int arrayDim = arrayType.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM) {
      throw new XMPexception("array dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    XobjList initArrayDescFuncArgs = Xcons.List(arrayDescId.getAddr(),
                                                templateObj.getDescId().Ref(),
                                                Xcons.IntConstant(arrayDim),
                                                arrayElmtTypeRef,
                                                Xcons.SizeOf(arrayElmtType));

    Vector<Ident> accIdVector = new Vector<Ident>(arrayDim);
    for (int i = 0; i < arrayDim; i++) {
      Ident accId = null;
      if (isLocalPragma) {
        accId = XMPlocalDecl.addObjectId2(XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i,
					  Xtype.unsignedlonglongType, parentBlock);
	if (isStaticDesc) accId.setStorageClass(StorageClass.STATIC);
      }
      else {
        accId = globalDecl.declStaticIdent(XMP.GTOL_PREFIX_ + "acc_" + arrayName + "_" + i,
                                           Xtype.unsignedlonglongType);
      }

      accIdVector.add(accId);
    }

    alignedArray = new XMPalignedArray(arrayName, arrayElmtType, (ArrayType)arrayType,
                                       arrayDim, accIdVector,
                                       arrayId, arrayDescId, arrayAddrId,
                                       templateObj);

    if (isLocalPragma){
      alignedArray.setIsLocal();
      alignedArray.setIsStaticDesc(isStaticDesc);
    }
    if (isParameter) alignedArray.setIsParameter();
    if (isPointer) alignedArray.setIsPointer();
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

    // check <align-source> list, <align-subscrip> list
    XobjList alignSourceList = (XobjList)alignDecl.getArg(1);
    XobjList alignSubscriptList = (XobjList)alignDecl.getArg(3);
    XobjList alignSubscriptVarList = (XobjList)alignSubscriptList.left();
    XobjList alignSubscriptExprList = (XobjList)alignSubscriptList.right();

    // check <align-source> list
    if(arrayType.getRef().getKind() == Xtype.POINTER){
      throw new XMPexception("Pointer to pointer \"" + arrayName + "\" can not be used.");
    }

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
        declNotAlignFunc(alignedArray, alignSourceIndex, globalDecl, isLocalPragma, pb);
      }
      else if (alignSource.equals(XMP.COLON)) {
        if (!XMPutil.hasElmt(alignSubscriptVarList, XMP.COLON)) {
          throw new XMPexception("cannot find ':' in <align-subscript> list");
        }

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
        if (XMPutil.countElmts(alignSourceList, alignSource) != 1) {
          throw new XMPexception("multiple '" + alignSource + "' indicated in <align-source> list");
        }

        if (XMPutil.hasElmt(alignSubscriptVarList, alignSource)) {
          if (XMPutil.countElmts(alignSubscriptVarList, alignSource) != 1) {
            throw new XMPexception("multiple '" + alignSource + "' indicated in <align-subscript> list");
          }

          int alignSubscriptIndex = XMPutil.getFirstIndex(alignSubscriptVarList, alignSource);
	  if (!isPointer)
	    declAlignFunc(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
			  alignSubscriptExprList.getArg(alignSubscriptIndex), globalDecl, isLocalPragma, pb);
	  else 
	    declAlignFunc_pointer(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
				  alignSubscriptExprList.getArg(alignSubscriptIndex), globalDecl, isLocalPragma, pb);
        }
        else {
          throw new XMPexception("cannot find '" + alignSource + "' in <align-subscript> list");
        }
      }

      alignSourceIndex++;
    }

    if (isPointer){
      //if (!isParameter) XMPlocalDecl.removeLocalIdent(pb, arrayName);
      return;
    }

    // check alignSubscriptVarList
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();

      if (alignSubscript.equals(XMP.ASTERISK) || alignSubscript.equals(XMP.COLON)) {
        break;
      }

      if (XMPutil.hasElmt(alignSourceList, alignSubscript)) {
        if (XMPutil.countElmts(alignSourceList, alignSubscript) != 1) {
          throw new XMPexception("no/multiple '" + alignSubscript + "' indicated in <align-source> list");
        }
      }
      else {
        throw new XMPexception("cannot find '" + alignSubscript + "' in <align-source> list");
      }
    }

    // add array size to args: do this after declAlignFunc()
    for (int i = 0; i < arrayDim; i++, arrayType = arrayType.getRef()) {
      long dimSize = arrayType.getArraySize();
      if (dimSize == 0) {
        throw new XMPexception("array size should be declared statically");
      } else if (dimSize == -1) {
        initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, arrayType.getArraySizeExpr()));
      } else {
        initArrayDescFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.LongLongConstant(0, dimSize)));
      }
    }

    // init array communicator
    XobjList initArrayCommFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
    for (XobjArgs i = alignSubscriptVarList.getArgs(); i != null; i = i.nextArgs()) {
      String alignSubscript = i.getArg().getString();

      if (alignSubscript.equals(XMP.ASTERISK)) {
        initArrayCommFuncArgs.add(Xcons.IntConstant(1));
      }
      else {
        initArrayCommFuncArgs.add(Xcons.IntConstant(0));
      }
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
						    arrayId.Ref(),
						    arrayDescId.Ref());
	for (int i = arrayDim - 1; i >= 0; i--) {
	  initArrayAddrFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
					       alignedArray.getAccIdAt(i).getAddr()));
	}

	XMPlocalDecl.addAllocCall2("_XMP_init_array_addr", initArrayAddrFuncArgs, globalDecl, parentBlock);
	XobjList bodyList = (XobjList)parentBlock.getProp("XCALABLEMP_PROP_LOCAL_ALLOC");
	if (isStaticDesc)
	  bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, Xcons.Set(alignedArray.getFlagId().Ref(), Xcons.IntConstant(1))));
      }
      else {

	XobjList allocFuncArgs = Xcons.List(alignedArray.getAddrIdVoidAddr(), alignedArray.getDescId().Ref());
	for (int i = alignedArray.getDim() - 1; i >= 0; i--) {
	  allocFuncArgs.add(Xcons.Cast(Xtype.Pointer(Xtype.unsignedlonglongType),
				       alignedArray.getAccIdAt(i).getAddr()));
	}

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
  }

  private static void declNotAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                                       XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {

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
                                   XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    Xobject templateLower = templateObj.getLowerAt(alignSubscriptIndex);
    Xobject alignNormExpr = Xcons.binaryOp(Xcode.MINUS_EXPR,
                                           alignSubscriptExpr, templateLower);
    alignedArray.setAlignNormExprAt(alignNormExpr, alignSourceIndex);

    // normalize 1. array size on src code: += normExpr
    alignedArray.normArraySize(alignSourceIndex, alignNormExpr);

    // normalize 2. alignSubscriptExpr: templateLower
    return templateLower;
  }

  private static void declAlignFunc(XMPalignedArray alignedArray, int alignSourceIndex,
                                    XMPtemplate templateObj, int alignSubscriptIndex,
                                    Xobject alignSubscriptExpr,
                                    XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {

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
    alignedArray.setAlignMannerAt(XMPalignedArray.convertDistMannerToAlignManner(distManner), alignSourceIndex);

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
        throw new XMPexception("unknown distribute manner");
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

  private static void declAlignFunc_pointer(XMPalignedArray alignedArray, int alignSourceIndex,
					    XMPtemplate templateObj, int alignSubscriptIndex,
					    Xobject alignSubscriptExpr,
					    XMPglobalDecl globalDecl, boolean isLocalPragma, PragmaBlock pb) throws XMPexception {

    Block parentBlock = null;
    if (isLocalPragma) parentBlock = pb.getParentBlock();

    // not normalize pointers. normalization should be done at runtime.
    //alignSubscriptExpr = normArray(alignedArray, alignSourceIndex, templateObj, alignSubscriptIndex,
    //                               alignSubscriptExpr, globalDecl, isLocalPragma, pb);

    XobjList alignFuncArgs = Xcons.List(alignedArray.getDescId().Ref(),
                                        Xcons.IntConstant(alignSourceIndex),
                                        Xcons.IntConstant(alignSubscriptIndex));

    alignFuncArgs.add(alignSubscriptExpr);

    int distManner = templateObj.getDistMannerAt(alignSubscriptIndex);
    alignedArray.setAlignMannerAt(XMPalignedArray.convertDistMannerToAlignManner(distManner), alignSourceIndex);

    alignedArray.setAlignSubscriptIndexAt(alignSubscriptIndex, alignSourceIndex);
    alignedArray.setAlignSubscriptExprAt(alignSubscriptExpr, alignSourceIndex);

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
