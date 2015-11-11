/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */

/*---------------------------------------*\
 * Adapter about array shape of Fortran
\*---------------------------------------*/

package exc.object;
import exc.block.Block;
import java.util.*;
import java.math.BigInteger;

public class Fshape
{
  private Block block = null;

  private int _rank;
  private Xobject[] _lbound;
  private Xobject[] _ubound;
  private Xobject[] _extent;

  //
  // constructors
  //
  public Fshape(FindexRange findexRange)
  {
    _rank = findexRange.getFrank();
    _lbound = findexRange.getLbounds();
    _ubound = findexRange.getUbounds();
    _extent = findexRange.getExtents();
  }

  public Fshape(FarrayType farrayType, Block block)
  {
    this(farrayType.getFindexRange(block));
    this.block = block;
  }

  public Fshape(Ident ident)
  {
    _rank = ident.getFrank();

    if (_rank == 0) {
      _lbound = null;
      _ubound = null;
      _extent = null;
      return;
    }

    Xtype type = ident.Type();
    switch (type.getKind()) {
    case Xtype.BASIC:
      _restrict("Ident of Xtype.BASIC not supported. ident=" + ident);
      break;

    case Xtype.F_ARRAY:
      Fshape fshape = new Fshape((FarrayType)type, block);
      _lbound = fshape.lbounds();
      _ubound = fshape.ubounds();
      _extent = fshape.extents();
      break;

    case Xtype.F_COARRAY:
      _restrict("Ident of Xtype.F_COARRAY not supported. ident=" + ident);
      // type.coshape might be useful.
      break;

    default:
      /* illegal kind of Xtype */
      _error("Found illegal type of ident. ident=" + ident);
      return;
    }
  }

  public Fshape(XobjString xobjString, Block block)
  {
    this(xobjString.findIdent(block));
  }
    

  public Fshape(Xobject xobj)
  {
    this(xobj, null);
  }

  public Fshape(Xobject xobj, Block block)
  {
    this.block = block;

    _error("unexpected Xobject. xobj=" + xobj);
  }

  //
  // access methods
  //
  public Xobject[] lbounds()
  {
    return _lbound;
  }
  public Xobject lbound(int dim)
  {
    return _lbound[dim];
  }

  public Xobject[] ubounds()
  {
    return _ubound;
  }
  public Xobject ubound(int dim)
  {
    return _ubound[dim];
  }

  public Xobject[] extents()
  {
    return _extent;
  }
  public Xobject extent(int dim)
  {
    return _extent[dim];
  }


  /*************************************************\
   * error handler
  \*************************************************/
  private static void _info(String msg) {
    System.err.println("[XMP Fshape] INFO: " + msg);
  }

  private static void _warn(String msg) {
    System.err.println("[XMP Fshape] WARNING: " + msg);
  }

  private static void _error(String msg) {
    System.err.println("[XMP Fshape] ERROR: " + msg);
  }

  private static void _restrict(String msg) {
    System.err.println("[XMP Fshape] RESTRICTION: " + msg);
  }

}

