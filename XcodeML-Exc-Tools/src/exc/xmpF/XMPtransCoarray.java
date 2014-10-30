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

  private FuncDefBlock procDef;
  private String procName;
  private BlockList procDecls;
  // - specification part of the procedure.
  // - contains (Xobject) id_list and (Xobject) decls.
  // - useful methods are:
  //     add/removeIdent, findLocalDecl(String)

  private Vector<XMPcoarray> coarrayList;
  private XMPinitProc initProc;

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
    initProc = new XMPinitProc(procDef);

    for (XMPcoarray coarray: coarrayList)
      transCoarrayDecl(coarray);
    ///    XMP.exitByError();   // exit if error has found.

    initProc.finalize();
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
    //     subroutine foo
    //       real var(10,20)[4,*]
    //   -------------------------------------------------------
    //   to ----------------------------------------------------
    //     subroutine foo
    //       real var(10,20)                                           ! (2)
    //       pointer (xmpf_craypointer_var, var)                       ! (3)
    //   and ---------------------------------------------------
    //     subroutine xmpf_init_foo   ! or xmpf_init_1host_name_foo
    //       call xmp_coarray_malloc(xmp_craypointer_var,  &
    //                               size(var), sizeof(var(1,1)))      ! (4)
    //   -------------------------------------------------------

    // (1) error check
    if (!coarray.isScalar() && !coarray.isExplicitShape())
      XMP.error("Coarray must be scalar or explicit shape: "+name);

    // (2) remove codimensions form coarray
    coarray.resetCodimensions();

    // (3) declaration of a cray pointer
    Ident cpIdent = procDecls.declLocalIdent("xmpf_craypointer_"+name,
                                             BasicType.Fint8Type,
                                             StorageClass.FLOCAL,
                                             Xcons.FvarRef(ident));  // ident.Ref() for C
    cpIdent.Type().setIsFcrayPointer(true);

    // (4) memory allocation
    //Xobject unit = ident.getFelementLengthExpr(); 
    Xobject unit = Xcons.IntConstant(4);      //**** TEMPORARY ****
    Xobject size = ident.Type().getFtotalSizeExpr();
    Xobject args = Xcons.List(Xcons.FvarRef(cpIdent), size, unit);
    Ident mallocIdent = procDecls.declLocalIdent("xmp_coarray_malloc",
                                                 BasicType.FsubroutineType);
    Xobject callStmt = Xcons.functionCall(mallocIdent, args);
    initProc.add(callStmt);

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
