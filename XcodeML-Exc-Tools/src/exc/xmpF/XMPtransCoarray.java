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
 * Translate Fortran Coarray
 */
public class XMPtransCoarray {

  final String CRAYPOINTER_PREFIX = "xmpf_cptr_";
  final String MALLOC_LIB_NAME = "xmp_coarray_malloc";
  final String COMMONBLOCK_NAME = "xmpf_cptr";

  private FuncDefBlock procDef;
  private String procName;
  private BlockList procDecls;
  // - specification part of the procedure.
  // - contains (Xobject) id_list and (Xobject) decls.
  // - useful methods are:
  //     add/removeIdent, findLocalDecl(String)

  private Vector<XMPcoarray> coarrayList;
  private XMPinitProcedure initProcedure;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPtransCoarray(FuncDefBlock procDef) {
    this.procDef = procDef;
    FunctionBlock fblock = procDef.getBlock();
    procName = fblock.getName();
    procDecls = fblock.getBody();

    // set all coarrays into coarrayList.
    setCoarrayList();
  }

  //------------------------------
  //  TRANSLATION
  //------------------------------
  public void run() {
    initProcedure = new XMPinitProcedure(procDef);

    for (XMPcoarray coarray: coarrayList)
      transCoarrayDecl(coarray);
    ///    XMP.exitByError();   // exit if error has found.

    initProcedure.finalize();
  }

  public void transCoarrayDecl(XMPcoarray coarray) {
    if (coarray.isPointer())
      XMP.error("Coarray cannot be a pointer: "+coarray.getName());
    else if (coarray.isAllocatable())
      transCoarrayDeclAllocatable(coarray);
    else 
      transCoarrayDeclStatic(coarray);
  }

  //------------------------------
  //  TRANSLATION (1) local/static coarray
  //------------------------------
  public void transCoarrayDeclStatic(XMPcoarray coarray) {
    String name = coarray.getName();
    Ident ident = coarray.getIdent();

    //   convert from ------------------------------------------
    //     subroutine fff
    //       real var(10,20)[4,*]
    //       ...
    //     end subroutine
    //   -------------------------------------------------------
    //   to ----------------------------------------------------
    //     subroutine fff
    //       real xxx(10,20)                                   ! (1)
    //       pointer (xmpf_cptr_xxx, xxx)                      ! (2)
    //       common /xmpf_cptr/...,xmpf_cptr_xxx,...           ! (4)
    //       ...
    //     end subroutine
    //     subroutine xmpf_init_unique_name_from_fff
    //       common /xmpf_cptr/...,xmpf_cptr_xxx,...           ! (4)
    //       call xmp_coarray_malloc(xmpf_cptr_xxx,200,4)      ! (3)
    //       ...
    //     end subroutine
    //   -------------------------------------------------------

    // (0) error check
    if (!coarray.isScalar() && !coarray.isExplicitShape())
      XMP.error("Coarray must be scalar or explicit shape: "+name);

    // (1) remove codimensions form coarray
    coarray.resetCodimensions();

    // (2) declaration of a cray pointer
    Ident cpIdent = procDecls.declLocalIdent(CRAYPOINTER_PREFIX + name,
                                             BasicType.Fint8Type,
                                             StorageClass.FLOCAL,
                                             Xcons.FvarRef(ident));  // ident.Ref() for C
    cpIdent.Type().setIsFcrayPointer(true);

    // (3) generate call stmt
    Xobject elem = ident.Type().getFelementLengthExpr(); 
    if (elem == null)
      XMP.error("Restriction: could not get the element length of: "
                + coarray.getName());
    if (!elem.isIntConstant())
      XMP.error("Restriction: could not evaluate the element length of: "
                + coarray.getName());

    Xobject count = ident.Type().getFnumElementsExpr();
    if (count == null)
      XMP.error("Restriction: could not get the number of elements of: "
                + coarray.getName());
    if (!count.isIntConstant())
      XMP.error("Restriction: could not evaluate the number of elements of: "
                + coarray.getName());

    /* call MALLOC_LIB_NAME(cpIdent,count,elem) */
    Ident mallocIdent = procDecls.declLocalIdent(MALLOC_LIB_NAME,
                                                 BasicType.FsubroutineType);
    Xobject args = Xcons.List(Xcons.FvarRef(cpIdent), count, elem);
    Xobject callStmt = Xcons.functionCall(mallocIdent, args);
    initProcedure.addStmt(callStmt);

    ///////////
    System.out.println(coarray.display());    
    ///////////

  }

  //------------------------------
  //  TRANSLATION (2) local/allocatable coarray
  //------------------------------
  public void transCoarrayDeclAllocatable(XMPcoarray coarray) {
    XMP.error("Allocatable coarry is not supported yet: "+coarray.getName());
  }


  //------------------------------
  //  TRANSLATION (3) module/static coarray
  //------------------------------


  //------------------------------
  //  TRANSLATION (4) module/allocatable coarray
  //------------------------------



  //------------------------------
  //  UTILITIES
  //------------------------------
  public Vector<XMPcoarray> getCoarrayList() {
    return coarrayList;
  }

  private void setCoarrayList() {
    coarrayList = new Vector();
    Xobject idList = procDef.getDef().getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.Type().getCorank() > 0)
        coarrayList.add(new XMPcoarray(ident));
    }
  }

  public String toString() {
    String s = "{";
    String delim = "";
    for (XMPcoarray coarray: coarrayList) {
      s += delim + coarray.toString();
      delim = ",";
    }
    return s + "}";
  }

  private String display() {
    String s = "";
    for (XMPcoarray coarray: coarrayList)
      s += coarray.display() + "\n";
    return s;
  }
}
