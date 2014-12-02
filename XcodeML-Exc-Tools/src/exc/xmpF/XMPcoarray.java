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

  // attributes
  private Ident ident;
  private String name;
  private Xtype originalType;

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
    //coshape = _getCoshape(
    originalType = ident.Type().copy();  // not sure how deep this copy
    if (DEBUG) System.out.println("[XMPcoarray] new coarray = "+this);
  }

  //------------------------------
  //  actions
  //------------------------------
  public void declareIdents(String crayPtrPrefix, String descrPrefix) {

    crayPtrName = crayPtrPrefix + "_" + name;
    descrName = descrPrefix + "_" + name;

    // declaration into fblock and set crayPtrId
    BlockList blist = fblock.getBody();

    // for descriptor (serial number of the coarray)
    descrId = blist.declLocalIdent(descrName,
                                   BasicType.FintType,
                                   StorageClass.FLOCAL,
                                   null);

    // for cray pointer
    crayPtrId = blist.declLocalIdent(crayPtrName,
                                     BasicType.Fint8Type,   // FintType ??
                                     StorageClass.FLOCAL,
                                     Xcons.FvarRef(ident));  // ident.Ref() for C
    crayPtrId.Type().setIsFcrayPointer(true);
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
    if (isPointer())
      XMP.error("Coarray cannot be a pointer: "+name);
    if (isAllocatable()) {
      //
    } else {
      if (!isScalar() && !isExplicitShape())
        XMP.error("Static coarray should be scalar or explicit shape: "+name);
    }
  }

  //------------------------------
  //  evaluation
  //------------------------------
  public int getElementLength() {
    Xobject elem = getElementLengthExpr(); 
    if (!elem.isIntConstant()) {
      XMP.error("Restriction: could not evaluate the element length of: "+name);
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
      XMP.error("Restriction: could not get the element length of: "+name);
    return elem;
  }

  public int getTotalArraySize() {
    Xobject size = getTotalArraySizeExpr();
    if (!size.isIntConstant()) {
      XMP.error("Restriction: could not evaluate the total size of: "+name);
      return 0;
    }
    return size.getInt();
  }
  public Xobject getTotalArraySizeExpr() {
    return getTotalArraySizeExpr(fblock);
  }
  public Xobject getTotalArraySizeExpr(Block block) {
    Xobject size = ident.Type().getTotalArraySizeExpr(block);
    if (size == null)
      XMP.error("Restriction: could not get the size of: "+name);
    return size;
  }

  public int getRank() {
    return ident.Type().getNumDimensions();
  }

  public Xobject getLbound(int i) {
    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getLbound(i, fblock);
  }

  public Xobject getUbound(int i) {
    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getUbound(i, fblock);
  }

  public Xobject getSizeFromIndexRange(Xobject range)
  {
    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getSizeFromIndexRange(range, fblock);
  }

  public Xobject getSizeFromLbUb(Xobject lb, Xobject ub)
  {
    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getSizeFromLbUb(lb, ub, fblock);
  }

  public Xobject getSizeFromTriplet(int i, Xobject i1, Xobject i2,
                                    Xobject i3) {
    FarrayType ftype = (FarrayType)ident.Type();
    return ftype.getSizeFromTriplet(i, i1, i2, i3, fblock);
  }

  //------------------------------
  //  inquiring interface
  //------------------------------
  public Boolean isScalar() {
    return (ident.Type().getNumDimensions() == 0);
  }

  public Boolean isAllocatable() {
    return ident.Type().isFallocatable();
  }

  public Boolean isPointer() {
    return ident.Type().isFpointer();
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

  public Xobject[] getCodimensions() {
    return ident.Type().getCodimensions();
  }

  public void setCodimensions(Xobject[] codimensions) {
    ident.Type().setCodimensions(codimensions);
  }

  public void removeCodimensions() {
    ident.Type().removeCodimensions();
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
}

