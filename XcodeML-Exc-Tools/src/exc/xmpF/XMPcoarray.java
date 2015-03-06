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

  // original attributes
  private Ident ident;
  private String name;
  private FindexRange indexRange;
  private Xtype originalType;
  private Boolean isAllocatableOriginally;
  private Boolean isPointerOriginally;

  // corresponding cray pointer and descriptor
  private String crayPtrName = null;
  private Ident crayPtrId = null;
  private String descrName = null;
  private Ident descrId = null;

  // context
  protected XMPenv env;
  protected XobjectDef def;
  protected FunctionBlock fblock;

  // for debug
  private Boolean DEBUG = false;        // switch me on debugger

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoarray(Ident ident, FuncDefBlock funcDef, XMPenv env) {
    this.ident = ident;
    this.env = env;
    def = funcDef.getDef();
    fblock = funcDef.getBlock();
    name = ident.getName();
    originalType = ident.Type().copy();  // not sure how deep this copy is
    isAllocatableOriginally = ident.Type().isFallocatable();
    isPointerOriginally = ident.Type().isFpointer();
    if (DEBUG) System.out.println("[XMPcoarray] new coarray = "+this);
  }

  //------------------------------
  //  actions
  //------------------------------

  // declare cray-pointer and decriptor correspoinding to the coarray.
  //
  public void declareIdents(String crayPtrPrefix, String descrPrefix) {

    crayPtrName = crayPtrPrefix + "_" + name;
    descrName = descrPrefix + "_" + name;

    // declaration into fblock and set crayPtrId
    BlockList blist = fblock.getBody();

    // for descriptor (the serial number of the coarray)
    descrId = blist.declLocalIdent(descrName,
                                   BasicType.FintType,
                                   StorageClass.FLOCAL,
                                   null);

    // for cray pointer
    Xtype crayPtrType = Xtype.Farray(BasicType.FintType);   // or Fint8Type ?
    crayPtrType.setIsFcrayPointer(true);
    crayPtrId = blist.declLocalIdent(crayPtrName,
                                     crayPtrType,
                                     StorageClass.FLOCAL,
                                     Xcons.FvarRef(ident));  // ident.Ref() if C
  }


  /*** not used now ***/
  public Xobject genMallocCallStmt(String mallocLibName) {
    BlockList blist = fblock.getBody();
    Ident mallocId = blist.declLocalIdent(mallocLibName,
                                          BasicType.FsubroutineType);
    Xobject varRef = Xcons.FvarRef(getCrayPointerId());
    Xobject elem = getElementLengthExpr(); 
    Xobject count = getTotalArraySizeExpr();
    Xobject args = Xcons.List(varRef, count, elem);
    Xobject stmt = Xcons.functionCall(mallocId, args);
    return stmt;
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
        if (!isScalar() && !isExplicitShape() && !isAllocatable())
          XMP.error("Coarray dummy argument must be explicit-shaped or allocatable: "
                    + name);
      }
    }

  }


  //------------------------------
  //  evaluation
  //------------------------------
  public int getElementLength() {
    Xobject elem = getElementLengthExpr(); 
    if (!elem.isIntConstant()) {
      XMP.error("current restriction: " +
                "could not numerically evaluate the element length of: "+name);
      return 0;
    }
    return elem.getInt();
  }

  public Xobject getElementLengthExpr() {
    return getElementLengthExpr(fblock);
  }
  public Xobject getElementLengthExpr(Block block) {
    Xobject elem = ident.Type().getElementLengthExpr(block);    // see BasicType.java
    if (elem == null)
      XMP.error("current restriction: " + 
                "could not find the element length of: "+name);
    return elem;
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
    Xobject size = getFindexRange().getTotalArraySizeExpr();
    if (size == null)
      XMP.error("current restriction: " +
                "could not find the total size of: "+name);
    return size;
  }

  /*********** not used 
  public Xobject getTotalArraySizeExpr(Block block) {
    //// wrong way because of lack of env
    //Xobject size = ident.Type().getTotalArraySizeExpr(block);
    ****/

  public int getRank() {
    return ident.Type().getNumDimensions();
  }


  public Xobject[] getShape() {
    if (getRank() == 0)
      return new Xobject[0];

    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getFarraySizeExpr();
  }

  private void _setFindexRange() {
    Xobject[] shape = getShape();
    indexRange = new FindexRange(shape, fblock, env);
  }

  private void _setFindexRange(Block block, XMPenv env) {
    Xobject[] sizes = getShape();
    indexRange = new FindexRange(sizes, block, env);
  }

  public FindexRange getFindexRange() {
    if (indexRange == null)
      _setFindexRange();
    return indexRange;
  }


  public Xobject getLbound(int i) {
    Xobject lbound = getLboundOrNull(i);
    if (lbound == null) {
      // generate intrinsic function call "lbound(a,dim)"
      Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident fname =
          getEnv().declIntrinsicIdent("lbound", Xtype.FintFunctionType);
      lbound = fname.Call(Xcons.List(arg1, arg2));
    }
    return lbound;
  }

  public Xobject getLboundOrNull(int i) {
    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getLbound(i, fblock);
  }

  public Xobject getUbound(int i) {
    Xobject ubound = getUboundOrNull(i);
    if (ubound == null) {
      // generate intrinsic function call "ubound(a,dim)"
      Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
      Xobject arg2 = Xcons.IntConstant(i + 1);
      Ident fname =
          getEnv().declIntrinsicIdent("ubound", Xtype.FintFunctionType);
      ubound = fname.Call(Xcons.List(arg1, arg2));
    }
    return ubound;
  }

  public Xobject getUboundOrNull(int i) {
    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getUbound(i, fblock);
  }

  public Xobject getSizeFromLbUb(Xobject lb, Xobject ub)
  {
    return getFindexRange().getSizeFromLbUb(lb, ub);
  }

  public Xobject getSizeFromIndexRange(Xobject range)
  {
    Xobject i1 = range.getArg(0);
    Xobject i2 = range.getArg(1);
    Xobject i3 = range.getArg(2);
    return getFindexRange().getSizeFromTriplet(i1, i2, i3);
  }


  //public Xobject getSizeFromTriplet(Xobject i1, Xobject i2, Xobject i3)
  //{
  //  return getFindexRange().getSizeFromTriplet(i1, i2, i3);
  //}

  public Xobject getSizeFromTriplet(int i, Xobject i1, Xobject i2, Xobject i3)
  {
    return getFindexRange().getSizeFromTriplet(i, i1, i2, i3);
  }


  //------------------------------
  //  get/set Xtype object
  //------------------------------
  public Boolean isScalar() {
    return (ident.Type().getNumDimensions() == 0);
  }

  public Boolean isAllocatable() {
    return isAllocatableOriginally;
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

  public Boolean isPointer() {
    return isPointerOriginally;
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
    return ident.getFdeclaredModule() != null;
  }

  public Ident getIdent() {
    return ident;
  }

  public XMPenv getEnv() {
    return env;
  }

  public String getCrayPointerName() {
    return crayPtrName;
  }

  public Ident getCrayPointerId() {
    return crayPtrId;
  }

  public String getDescriptorName() {
    return descrName;
  }

  public Ident getDescriptorId() {
    return descrId;
  }

  public Xobject getDescriptorIdExpr(Xobject baseAddr) {
    if (descrId != null)
      return descrId;

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

  public int getCorank() {
    return ident.Type().getCorank();
  }

  public String getName() {
    return ident.getName();
  }

  public Xtype getType() {
    return ident.Type();
  }

  public Xtype getOriginalType() {
    return originalType;
  }

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

  public String display() {
    return "{ident:" + toString(ident)
      + ", originalType:" + toString(originalType)
      + "}";
  }


  //------------------------------------------------------------
  //  Fortran Type and Kind
  //   ******** under construction *********
  //------------------------------------------------------------

  /* TEMPORARY VERSION
   *   getFkind().getKind() is useful, which is integer as Xtype.Fxxx
   */
  public Xtype getFtype() {
    Xtype ftype = ident.Type();
    if (ftype.getKind() == Xtype.F_ARRAY)
      ftype = ftype.getRef();
    return ftype;
  }

  /* TEMPORARY VERSION
   *   getFkind().getInt() is useful.
   */
  public Xobject getFkind() {
    return getFtype().getFkind();
  }

  /*
   * return a name of Fortran intrinsic function
   */
  private String getTypeIntrinsicName() {
    return getTypeIntrinsicName(getType());
  }

  private String getTypeIntrinsicName(Xtype xtype) {
    String name = null;

    switch (xtype.getKind()) {
    case Xtype.F_ARRAY:
      name = _getTypeIntrinName_1(xtype.getRef());
      break;
    case Xtype.BASIC:
      name = _getTypeIntrinName_1(xtype);
      break;
    case Xtype.STRUCT:
      //XMP.error("internal error: STRUCT unsupported in _getTypeSuffix()");
    default:
      //XMP.error("internal error: unexpected kind in _getTypeSuffix(): xtype.getKind()");
      break;
    }

    return name;
  }

  /// see also BasicType.getElementLength
  private String _getTypeIntrinName_1(Xtype xtype) {
    String key = null;
    switch(xtype.getBasicType()) {
    case BasicType.BOOL:
      key = "l";
      break;
    case BasicType.SHORT:
    case BasicType.UNSIGNED_SHORT:
    case BasicType.INT:
    case BasicType.UNSIGNED_INT:
    case BasicType.LONG:
    case BasicType.UNSIGNED_LONG:
    case BasicType.LONGLONG:
    case BasicType.UNSIGNED_LONGLONG:
      key = "int";
      break;
    case BasicType.FLOAT:
    case BasicType.DOUBLE:
    case BasicType.LONG_DOUBLE:
      key = "real";
      break;
    case BasicType.FLOAT_COMPLEX:
    case BasicType.DOUBLE_COMPLEX:
    case BasicType.LONG_DOUBLE_COMPLEX:
      key = "cmplx";
      break;
    case BasicType.CHAR:
    case BasicType.UNSIGNED_CHAR:
    case BasicType.F_CHARACTER:
      key = "c";
      break;

    default:
      XMP.error("found unsupported type of coarray");
      break;
    }

    return key;
  }

}

