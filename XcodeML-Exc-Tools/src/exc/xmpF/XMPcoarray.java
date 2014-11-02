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

  final String CRAYPOINTER_PREFIX = "xmpf_cptr_";
  final String MALLOC_LIB_NAME = "xmp_coarray_malloc";

  private String name;
  private Ident ident;
  private BlockList decls;

  private Xtype originalType;
  private String crayPointerName;
  private Ident crayPointerId;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoarray(Ident ident, BlockList decls) {
    this.ident = ident;
    this.decls = decls;
    name = ident.getName();
    originalType = ident.Type().copy();  // not sure how deep this copy
    crayPointerName = CRAYPOINTER_PREFIX + name;
    crayPointerId = null;
  }

  //------------------------------
  //  actions about malloc coarray
  //------------------------------
  public void declareCrayPointer() {
    if (crayPointerId != null)
      XMP.error("Internal error: crayPointerId has already declared.");

    // both declaration into decls and set crayPointerId
    crayPointerId = decls.declLocalIdent(crayPointerName,
                                         BasicType.Fint8Type,
                                         StorageClass.FLOCAL,
                                         Xcons.FvarRef(ident));  // ident.Ref() for C
    crayPointerId.Type().setIsFcrayPointer(true);
  }

  public Xobject genMallocCallStmt() {
      Ident mallocId = decls.declLocalIdent(MALLOC_LIB_NAME,
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
  //  inquiring interface
  //------------------------------
  public Xobject getElementLengthExpr() {
    return getElementLengthExpr(decls);
  }
  public Xobject getElementLengthExpr(BlockList decls) {
    Xobject elem = ident.Type().getElementLengthExpr(decls); 
    if (elem == null)
      XMP.error("Restriction: could not get the element length of: "+name);
    if (!elem.isIntConstant())
      XMP.error("Restriction: could not evaluate the element length of: "+name);
    return elem;
  }

  public int getElementLength() {
    return getElementLength(decls);
  }
  public int getElementLength(BlockList decls) {
    return getElementLengthExpr(decls).getInt();
  }

  public Xobject getTotalArraySizeExpr() {
    return getTotalArraySizeExpr(decls);
  }
  public Xobject getTotalArraySizeExpr(BlockList decls) {
    Xobject size = ident.Type().getTotalArraySizeExpr(decls);
    if (size == null)
      XMP.error("Restriction: could not get the size of: "+name);
    return size;
  }

  public int getTotalArraySize() {
    return getTotalArraySize(decls);
  }
  public int getTotalArraySize(BlockList decls) {
    Xobject size = getTotalArraySizeExpr(decls);
    if (!size.isIntConstant()) {
      XMP.error("Restriction: could not evaluate the total size of: "+name);
      return 0;
    }
    return size.getInt();
  }

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

  public String getCrayPointerName() {
    return crayPointerName;
  }

  public void setCrayPointerId(String crayPointerName) {
    this.crayPointerName = crayPointerName;
  }

  /* crayPointerId is set only in declareCrayPointer()
   */

  public Ident getCrayPointerId() {
    return crayPointerId;
  }

  public Xobject[] getCodimensions() {
    return ident.Type().getCodimensions();
  }

  public void setCodimensions(Xobject[] codimensions) {
    ident.Type().setCodimensions(codimensions);
  }

  public void clearCodimensions() {
    ident.Type().clearCodimensions();
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

