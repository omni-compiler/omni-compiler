/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Madiator for each coarray
 */
public class XMPcoarray {

  // name of library
  public final static String VAR_DESCPOINTER_PREFIX = "xmpf_descptr";
  public final static String VAR_CRAYPOINTER_PREFIX = "xmpf_crayptr";
  final static String XMPF_LCOBOUND = "xmpf_lcobound";
  final static String XMPF_UCOBOUND = "xmpf_ucobound";
  final static String XMPF_COSIZE = "xmpf_cosize";
  final static String GET_IMAGE_INDEX_NAME = "xmpf_coarray_get_image_index";
  final static String SET_COSHAPE_NAME = "xmpf_coarray_set_coshape";
  final static String SET_VARNAME_NAME = "xmpf_coarray_set_varname";

  // original attributes
  private Ident ident;
  private String name;
  private FindexRange indexRange = null;
  private FindexRange coindexRange = null;
  //private Xtype originalType;
  private Boolean isAllocatable;
  private Boolean isPointer;
  private Boolean isUseAssociated;
  private Boolean _wasMovedFromModule = false;

  // corresponding cray pointer, descriptor and common block names
  private String _crayPtrName = null;
  private Ident crayPtrId = null;
  private String _descPtrName = null;
  private Ident descPtrId = null;
  private String homeBlockName = null;
  
  // context
  protected XMPenv env;
  protected XobjectDef def;
  protected FunctionBlock fblock;

  // for debug
  private Boolean DEBUG = false;        // switch me on debugger

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoarray(Ident ident, FuncDefBlock funcDef, XMPenv env)
  {
    this(ident, funcDef.getDef(), funcDef.getBlock(), env);
  }
  public XMPcoarray(Ident ident, FuncDefBlock funcDef, XMPenv env,
                    String homeBlockName)
  {
    this(ident, funcDef.getDef(), funcDef.getBlock(), env, homeBlockName);
  }
  public XMPcoarray(Ident ident, XobjectDef def, FunctionBlock fblock, XMPenv env)
  {
    this.env = env;
    this.def = def;
    this.fblock = fblock;
    setIdent(ident);
    homeBlockName = ident.getFdeclaredModule();
    if (homeBlockName == null)
      homeBlockName = def.getName();
  }
  public XMPcoarray(Ident ident, XobjectDef def, FunctionBlock fblock, XMPenv env,
                    String homeBlockName)
  {
    this.env = env;
    this.def = def;
    this.fblock = fblock;
    setIdent(ident);
    this.homeBlockName = homeBlockName;
  }


  //------------------------------
  //  actions
  //------------------------------

  // declare cray-pointer variable correspoinding to this.
  //
  public void genDecl_crayPointer() {
    BlockList blist = fblock.getBody();
    String crayPtrName = getCrayPointerName();

    // generate declaration of crayPtrId
    Xtype crayPtrType = Xtype.Farray(BasicType.Fint8Type);
    crayPtrType.setIsFcrayPointer(true);
    crayPtrId = blist.declLocalIdent(crayPtrName,
                                     crayPtrType,
                                     StorageClass.FLOCAL,
                                     Xcons.FvarRef(ident));  // ident.Ref() if C
  }


  // declare variable of descriptor pointer corresponding to this.
  //
  public void genDecl_descPointer() {
    if(descPtrId != null) {
      return;
    }

    String descPtrName = getDescPointerName();
    BlockList blist = fblock.getBody();
    
    descPtrId = env.declInternIdent(descPtrName,
                                    BasicType.Fint8Type);
  }



  /*
   *  m. "CALL set_coshape(descPtr, corank, clb1, clb2, ..., clbr)"
   *     returns null if it is not allocated
   */
  public Xobject makeStmt_setCoshape() {
    return makeStmt_setCoshape(env);
  }

  public Xobject makeStmt_setCoshape(XMPenv env) {
    int corank = getCorank();

    Xobject args = Xcons.List(getDescPointerId(),
                              Xcons.IntConstant(corank));
    for (int i = 0; i < corank - 1; i++) {
      args.add(getLcobound(i));
      args.add(getUcobound(i));
    }
    args.add(getLcobound(corank - 1));
    if (args.hasNullArg())
      XMP.fatal("generated null argument " + SET_COSHAPE_NAME +
                "(makeStmt_setCoshape())");

    Ident subr = env.findVarIdent(SET_COSHAPE_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_COSHAPE_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  /*
   *  m. "CALL set_coshape(descPtr, corank, clb1, clb2, ..., clbr)"
   *     with static coshape
   */
  public Xobject makeStmt_setCoshape(XobjList coshape) {
    int corank = getCorank();
    if (corank != coshape.Nargs()) {
      XMP.fatal("number of codimensions not matched with the declaration:"
                + corank + " and " + coshape.Nargs());
      return null;
    }

    Xobject args = Xcons.List(getDescPointerId(),
                              Xcons.IntConstant(corank));
    for (int i = 0; i < corank - 1; i++) {
      args.add(_getLboundInIndexRange(coshape.getArg(i)));
      args.add(_getUboundInIndexRange(coshape.getArg(i)));
    }
    args.add(_getLboundInIndexRange(coshape.getArg(corank - 1)));
    if (args.hasNullArg())
      XMP.fatal("generated null argument " + SET_COSHAPE_NAME + 
                "(makeStmt_setCoshape(coshape))");

    Ident subr = env.findVarIdent(SET_COSHAPE_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_COSHAPE_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  private Xobject _getLboundInIndexRange(Xobject dimension) {
    Xobject lbound;

    if (dimension == null)
      lbound = null;
    else {
      switch (dimension.Opcode()) {
      case F_INDEX_RANGE:
        lbound = dimension.getArg(0);
        break;
      case F_ARRAY_INDEX:
        lbound = null;
        break;
      default:
        lbound = null;
        break;
      }
    }

    if (lbound == null)
      return Xcons.IntConstant(1);

    return lbound.cfold(fblock);
  }


  private Xobject _getUboundInIndexRange(Xobject dimension) {
    Xobject ubound;

    if (dimension == null)
      ubound = null;
    else {
      switch (dimension.Opcode()) {
      case F_INDEX_RANGE:
        ubound = dimension.getArg(1);
        break;
      case F_ARRAY_INDEX:
        ubound = dimension.getArg(0);
        break;
      default:
        ubound = dimension;
      }
    }

    if (ubound == null)
      XMP.fatal("illegal upper bound specified in ALLOCATE statement");

    return ubound.cfold(fblock);
  }


  /*
   *  n. "CALL set_varname(descPtr, name, namelen)"
   */
  public Xobject makeStmt_setVarName() {
    return makeStmt_setVarName(env);
  }

  public Xobject makeStmt_setVarName(XMPenv env) {
    String varName = getName();
    Xobject varNameObj = 
      Xcons.FcharacterConstant(Xtype.FcharacterType, varName, null);
    Xobject varNameLen = 
      Xcons.IntConstant(varName.length());
    Xobject args = Xcons.List(getDescPointerId(),
                              varNameObj, varNameLen);
    if (args.hasNullArg())
      XMP.fatal("generated null argument " + SET_VARNAME_NAME +
                "(makeStmt_setVarName)");

    Ident subr = env.findVarIdent(SET_VARNAME_NAME, null);
    if (subr == null) {
      subr = env.declExternIdent(SET_VARNAME_NAME,
                                 BasicType.FexternalSubroutineType);
    }
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  //------------------------------
  //  self error check
  //------------------------------
  public void errorCheck() {

    if (ident.isCoarray()) {  // if it is not converted yet
      if (isPointer()) {
        XMP.error("Coarray variable cannot be a pointer: " + name);
      }
      if (isDummyArg()) {
        if (isScalar() || isExplicitShape() || isAssumedSize() ||
            isAssumedShape() || isAllocatable())
          ;
        else
          XMP.error("Coarray dummy argument must be of explicit shape, assumed size, assumed shape, or allocatable: "
                    + name);
      }
    }

  }


  //------------------------------
  //  IndexRange (to be abolished)
  //------------------------------

  private void _setIndexRange() {
    Xobject[] shape = getShape();
    indexRange = new FindexRange(shape, fblock, env);
  }

  private void _setIndexRange(Block block, XMPenv env) {
    Xobject[] shape = getShape();
    indexRange = new FindexRange(shape, block, env);
  }

  public FindexRange getIndexRange() {
    if (indexRange == null)
      _setIndexRange();
    return indexRange;
  }


  //------------------------------
  //  CoindexRange
  //------------------------------

  private void _setCoindexRange() {
    Xobject[] shape = getCoshape();
    coindexRange = new FindexRange(shape, fblock, env);
  }

  private void _setCoindexRange(Block block, XMPenv env) {
    Xobject[] shape = getCoshape();
    coindexRange = new FindexRange(shape, block, env);
  }

  public FindexRange getCoindexRange() {
    if (coindexRange == null)
      _setCoindexRange();
    return coindexRange;
  }


  //------------------------------
  //  evaluate index
  //------------------------------
  public int getElementLengthOrNot() {
    Xobject elem = getElementLengthExpr(); 
    if (elem == null || !elem.isIntConstant())
      return -1;
    return elem.getInt();
  }

  public int getElementLength() {
    int elem = getElementLengthOrNot(); 
    if (elem < 0) {
      XMP.fatal("current restriction: " +
                "could not numerically evaluate the element length of: "+name);
    }
    return elem;
  }

  public Xobject getElementLengthExpr() {
    return getElementLengthExpr(fblock);
  }
  public Xobject getElementLengthExpr(Block block) {
    Xobject elem = ident.Type().getElementLengthExpr(block);    // see BasicType.java
    if (elem != null)
      return elem;

    // The element length was not detected from the Ident.

    if (getRank() == 0) {    // scalar coarray
      // copy type
      // size(transfer(ident, (/" "/))
      Ident sizeId = declIntIntrinsicIdent("size");
      Ident transferId = declIntIntrinsicIdent("transfer");
      Xobject arg1 = Xcons.FvarRef(ident);
      Xobject arg21 = Xcons.FcharacterConstant(Xtype.FcharacterType, " ", null);
      Xobject arg2 = Xcons.List(Xcode.F_ARRAY_CONSTRUCTOR,
                                _getCharFarrayType(1),
                                arg21);
      Xobject transfer = transferId.Call(Xcons.List(arg1, arg2));
      Xobject size = sizeId.Call(Xcons.List(transfer));
      return size;
    } else {                 // array coarray
    }

    return null;
  }

  public int getTotalArraySize() {
    Xobject size = getTotalArraySizeExpr();
    if (!size.isIntConstant()) {
      XMP.error("current restriction: " +
                "could not numerically evaluate the total size of: "+name);
      return 0;
    }
    return size.getInt();
  }

  public Xobject getTotalArraySizeExpr() {
    Xobject size = getIndexRange().getTotalArraySizeExpr();
    if (size == null)
      XMP.error("current restriction: " +
                "could not find the total size of: "+name);
    return size;
  }


  //------------------------------
  //  inquire in Fortran terminology:
  //   rank, shape, lower/upper bound and size
  //------------------------------

  public int getRank() {
    return ident.Type().getNumDimensions();
  }

  public Xobject[] getShape() {
    if (getRank() == 0)
      return new Xobject[0];

    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getFarraySizeExpr();
  }


  public Xobject getLboundStatic(int i) {
    if (isExplicitShape()) {
      FarrayType ftype = (FarrayType)ident.Type();
      return ftype.getLbound(i, fblock);
    }
    return null;
  }

  public Xobject getUboundStatic(int i) {
    if (isExplicitShape()) {
      FarrayType ftype = (FarrayType)ident.Type();
      return ftype.getUbound(i, fblock);
    }
    return null;
  }

  public Xobject getLbound(int i) {
    Xobject lbound = getLboundStatic(i);
    if (lbound == null) {
      // generate intrinsic function call "lbound(a,dim)"
      Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident lboundId = declIntIntrinsicIdent("lbound");
      lbound = lboundId.Call(Xcons.List(arg1, arg2));
    }
    return lbound;
  }

  public Xobject getUbound(int i) {
    Xobject ubound = getUboundStatic(i);
    if (ubound == null) {
      // generate intrinsic function call "ubound(a,dim)"
      Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident uboundId = declIntIntrinsicIdent("ubound");
      ubound = uboundId.Call(Xcons.List(arg1, arg2));
    }
    return ubound;
  }


  public Xobject getSizeFromLbUb(Xobject lb, Xobject ub) {
    return getIndexRange().getSizeFromLbUb(lb, ub);
  }

  public Xobject getSizeFromIndexRange(Xobject range) {
    Xobject i1 = range.getArg(0);
    Xobject i2 = range.getArg(1);
    Xobject i3 = range.getArg(2);
    return getIndexRange().getSizeFromTriplet(i1, i2, i3);
  }


  //public Xobject getSizeFromTriplet(Xobject i1, Xobject i2, Xobject i3)
  //{
  //  return getIndexRange().getSizeFromTriplet(i1, i2, i3);
  //}

  public Xobject getSizeFromTriplet(int i, Xobject i1, Xobject i2, Xobject i3) {
    return getIndexRange().getSizeFromTriplet(i, i1, i2, i3);
  }


  //------------------------------
  //  evaluation in Fortran terminology:
  //   corank, coshape, lower/upper cobound and cosize
  //------------------------------

  public int getCorank() {
    return ident.Type().getCorank();
  }
 
  public Xobject[] getCoshape() {
    return ident.Type().getCodimensions();
  }

  public Xobject getLcoboundStatic(int i) {
    FindexRange indexRange = getCoindexRange();
    return (indexRange == null) ? null : indexRange.getLbound(i);
  }

  public Xobject getUcoboundStatic(int i) {
    FindexRange indexRange = getCoindexRange();
    return (indexRange == null) ? null : indexRange.getUbound(i);
  }

  public Xobject getCosizeStatic(int i) {
    FindexRange indexRange = getCoindexRange();
    return (indexRange == null) ? null : indexRange.getExtent(i);
  }

  public Xobject getLcobound(int i) {
    Xobject lcobound = getLcoboundStatic(i);
    if (lcobound == null) {
      // generate intrinsic function call "xmpf_lcobound(serno, dim)"
      Xobject arg1 = descPtrId;
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident lcoboundId = getEnv().findVarIdent(XMPF_LCOBOUND, null);
      if (lcoboundId == null)
        lcoboundId = getEnv().declExternIdent(XMPF_LCOBOUND, Xtype.FintFunctionType);
      lcobound = lcoboundId.Call(Xcons.List(arg1, arg2));
    }
    return lcobound;
  }

  public Xobject getUcobound(int i) {
    Xobject ucobound = getUcoboundStatic(i);
    if (ucobound == null) {
      // generate intrinsic function call "xmpf_ucobound(serno, dim)"
      Xobject arg1 = descPtrId;
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident ucoboundId = getEnv().findVarIdent(XMPF_UCOBOUND, null);
      if (ucoboundId == null)
        ucoboundId = getEnv().declExternIdent(XMPF_UCOBOUND, Xtype.FintFunctionType);
      ucobound = ucoboundId.Call(Xcons.List(arg1, arg2));
    }
    return ucobound;
  }

  public Xobject getCosize(int i) {
    Xobject cosize = getCosizeStatic(i);
    if (cosize == null) {
      // generate intrinsic function call "xmpf_cosize(serno, dim)"
      Xobject arg1 = descPtrId;
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident cosizeId = getEnv().findVarIdent(XMPF_COSIZE, null);
      if (cosizeId == null)
        cosizeId = getEnv().declExternIdent(XMPF_COSIZE, Xtype.FintFunctionType);
      cosize = cosizeId.Call(Xcons.List(arg1, arg2));
    }
    return cosize;
  }


  //------------------------------
  //  evaluation in Fortran terminology:
  //   image index
  //------------------------------

  public Xobject getImageIndex(Xobject baseAddr, Xobject cosubscripts) {
    String fname = GET_IMAGE_INDEX_NAME;
    Ident fnameId = getEnv().findVarIdent(fname, null);
    if (fnameId == null)
      fnameId = getEnv().declExternIdent(fname, Xtype.FintFunctionType);

    XobjList args = Xcons.List(getDescPointerIdExpr(baseAddr), 
                               Xcons.IntConstant(getCorank()));
    for (Xobject cosubs: (XobjList)cosubscripts) {
      args.add(cosubs);
    }
    if (args.hasNullArg())
      XMP.fatal("generated null argument " + fname + "(getImageIndex)");

    return fnameId.Call(args);
  }



  //------------------------------
  //  tool
  //------------------------------
  private Ident declIntIntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident ident = getEnv().declIntrinsicIdent(name, ftype);
    return ident;
  }


  //------------------------------
  //  get/set Xtype object
  //------------------------------
  public Boolean isScalar() {
    return (ident.Type().getNumDimensions() == 0);
  }

  public Boolean isAllocatable() {
    return isAllocatable;
  }

  public void setAllocatable() {
    ident.Type().setIsFallocatable(true);
  }

  public void resetAllocatable() {
    for (Xtype type = ident.Type(); type != null; ) {
      type.setIsFallocatable(false);
      if (type.copied != null)
        type = type.copied;
      else if (type.isBasic())
        break;
      else
        type = type.getRef();
    }
  }

  public void resetSaveAttr() {
    Xtype type = ident.Type();
    _resetSaveAttrInType(type);
  }

  private void _resetSaveAttrInType(Xtype type) {
    type.setIsFsave(false);

    if (type.copied != null) 
      _resetSaveAttrInType(type.copied);

    if (type.isArray() || type.isFarray())
      _resetSaveAttrInType(type.getRef());
  }

  public Boolean isPointer() {
    return isPointer;
  }

  public void setPointer() {
    ident.Type().setIsFpointer(true);
  }

  public void resetPointer() {
    for (Xtype type = ident.Type(); type != null; ) {
      type.setIsFpointer(false);
      if (type.copied != null)
        type = type.copied;
      else if (type.isBasic())
        break;
      else
        type = type.getRef();
    }
  }

  public Boolean isDummyArg() {
    if (ident.getStorageClass() == StorageClass.FPARAM)
      return true;
    return false;
  }

  public Boolean isAssumedSize() {
    return ident.Type().isFassumedSize();
  }

  public Boolean isAssumedShape() {
    return ident.Type().isFassumedShape();
  }

  public Boolean isExplicitShape() {
    return (!isAssumedSize() && !isAssumedShape() &&
            !isAllocatable() && !isPointer());
  }

  public Boolean isUseAssociated() {
    return isUseAssociated;
  }



  public Ident getIdent() {
    return ident;
  }

  public void setIdent(Ident ident) {
    this.ident = ident;
    name = ident.getName();

    isAllocatable = ident.Type().isFallocatable();
    isPointer = ident.Type().isFpointer();
    isUseAssociated = (ident.getFdeclaredModule() != null);
  }

  public XobjectDef getDef() {
    return def;
  }

  public FunctionBlock getFblock() {
    return fblock;
  }

  public XMPenv getEnv() {
    return env;
  }

  public String getHomeBlockName()
  {
    return homeBlockName;
  }

  public String getDescCommonName()
  {
    return VAR_DESCPOINTER_PREFIX + "_" + homeBlockName;
  }

  public String getCrayCommonName()
  {
    return VAR_CRAYPOINTER_PREFIX + "_" + homeBlockName;
  }

  public String getCrayPointerName() {
    if (_crayPtrName == null) {
      _crayPtrName = VAR_CRAYPOINTER_PREFIX + "_" + name;
    }
    return _crayPtrName;
  }

  public Ident getCrayPointerId() {
    return crayPtrId;
  }

  public String getDescPointerName() {
    if (_descPtrName == null) {
      _descPtrName = VAR_DESCPOINTER_PREFIX + "_" + name;
    }

    return _descPtrName;
  }

  public Ident getDescPointerId() {
    if (descPtrId == null)
      XMP.warning("INTERNAL: illeagal null descPtrId");

    return descPtrId;
  }


  /*************** should be deleted .....
  ***************************/
  public Xobject getDescPointerIdExpr(Xobject baseAddr) {
    if (descPtrId != null)
      return descPtrId;

    Ident funcIdent =
      getEnv().declExternIdent("xmpf_get_descr_id", Xtype.FintFunctionType);
    Xobject descId = funcIdent.Call(Xcons.List(baseAddr));
    return descId;
  }

  public Xobject[] getCodimensions() {
    Xobject[] codims = ident.Type().getCodimensions();
    return codims;
  }

  public void setCodimensions(Xobject[] codimensions) {
    ident.Type().setCodimensions(codimensions);
  }

  public void removeCodimensions() {
    ident.Type().removeCodimensions();
  }

  public void hideCodimensions() {
    ident.Type().setIsCoarray(false);
  }

  public String getName() {
    return ident.getName();
  }

  public Xtype getType() {
    return ident.Type();
  }

  public void setWasMovedFromModule(Boolean bool) {
    _wasMovedFromModule = bool;
  }

  public Boolean wasMovedFromModule() {
    return _wasMovedFromModule;
  }


  //public Xtype getOriginalType() {
  //return originalType;
  //}

  public String toString() {
    return toString(ident);
  }
  public String toString(Xobject obj) {
    return "Xobject(" + obj.getName()
      + ",rank=" + obj.Type().getNumDimensions()
      + ",corank=" + obj.Type().getCorank()
      + ")";
  }
  public String toString(Xtype type) {
    return "Xtype(rank=" + type.getNumDimensions()
      + ",corank=" + type.getCorank()
      + ")";
  }



  //------------------------------
  //  low-level handling (NOT USED)
  //------------------------------
  public Ident unlinkIdent() {
    return unlinkIdent(def);
  }
  public Ident unlinkIdent(XobjectDef def) {
    return unlinkIdent((XobjList)def.getDef());
  }
  public Ident unlinkIdent(XobjList def) {
    XobjArgs args0 = def.getIdentList().getArgs();
    XobjArgs lastArgs = null;
    XobjArgs thisArgs = null;
    for (XobjArgs args = args0; args != null; args = args.nextArgs()) {
      Xobject arg = args.getArg();
      Ident id = (Ident)arg;
      if (id == ident) {
        thisArgs = args;
        break;
      }
      if (id.getName().equals(name)) {
        XMP.fatal("unexpected matching of ident names: " + name);
        thisArgs = args;
        break;
      }
      lastArgs = args;
    }

    if (thisArgs == null)   // not found
      return null;

    // unlink and reconnect
    if (lastArgs == null)
      def.getIdentList().setArgs(thisArgs.nextArgs());
    else
      lastArgs.setNext(thisArgs.nextArgs());

    thisArgs.setNext(null);

    return (Ident)thisArgs.getArg();
  }


  //------------------------------------------------------------
  //  Fortran Type and Kind
  //   ******** under construction *********
  //------------------------------------------------------------

  private Xtype _getXtype() {
    Xtype xtype = ident.Type();
    if (xtype.getKind() == Xtype.F_ARRAY)
      xtype = xtype.getRef();
    return xtype;
  }

  public int getFtypeNumber() {
    return _getXtype().getBasicType();
  }

  public Xobject getFkind() {
    return _getXtype().getFkind();
  }

  /*
   * return a name of Fortran intrinsic function
   */
  public String getFtypeString() {
    return _getTypeIntrinName_1(getFtypeNumber());
  }
  public String getFtypeString(int typeNumber) {
    return _getTypeIntrinName_1(typeNumber);
  }


  private Xtype _getCharFarrayType(int size) {
    Xtype ref = Xtype.FcharacterType;
    Xtype type = Xtype.Farray(ref, Xcons.IntConstant(size));
    return type;
  }

  /// see also BasicType.getElementLength
  private String _getTypeIntrinName_1(int typeNumber) {
    String tname = null;

    switch(typeNumber) {
    case BasicType.BOOL:
      tname = "logical";
      break;

    case BasicType.SHORT:
    case BasicType.UNSIGNED_SHORT:
    case BasicType.INT:
    case BasicType.UNSIGNED_INT:
    case BasicType.LONG:
    case BasicType.UNSIGNED_LONG:
    case BasicType.LONGLONG:
    case BasicType.UNSIGNED_LONGLONG:
      tname = "int";
      break;

    case BasicType.FLOAT:
    case BasicType.DOUBLE:
    case BasicType.LONG_DOUBLE:
      tname = "real";
      break;

    case BasicType.FLOAT_COMPLEX:
    case BasicType.DOUBLE_COMPLEX:
    case BasicType.LONG_DOUBLE_COMPLEX:
      tname = "cmplx";
      break;

    case BasicType.CHAR:
    case BasicType.UNSIGNED_CHAR:
    case BasicType.F_CHARACTER:
      tname = "char";
      break;

    case BasicType.F_NUMERIC_ALL:
      tname = null;
      break;

    default:
      XMP.fatal("found illegal type number in BasicType: " + typeNumber);
      break;
    }

    return tname;
  }

}

