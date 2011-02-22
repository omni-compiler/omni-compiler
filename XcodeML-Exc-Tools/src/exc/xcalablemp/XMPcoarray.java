/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import exc.block.*;
import java.util.Vector;

public class XMPcoarray {
  public final static int ACC_BIT_XOR	= 1;

  private String		_name;
  private Xtype			_type;
  private int			_dim;
  private Vector<Integer>	_sizeVector;
  private Ident			_descId;
  private Ident			_arrayId;
  private boolean		_isArray;
  private int			_winId;

  public XMPcoarray(String name, Xtype type, int dim, Vector<Integer> sizeVector,
                    Ident descId, Ident arrayId, boolean isArray, int winId) {
    _name = name;
    _type = type;
    _dim = dim;
    _sizeVector = sizeVector;
    _descId = descId;
    _arrayId = arrayId;
    _isArray = isArray;
    _winId = winId;
  }

  public String getName() {
    return _name;
  }

  public Xtype getType() {
    return _type;
  }

  public int getDim() {
    return _dim;
  }

  public int getSizeAt(int index) {
    return _sizeVector.get(index).intValue();
  }

  public Ident getDescId() {
    return _descId;
  }

  public Ident getArrayId() {
    return _arrayId;
  }

  public boolean isArray() {
    return _isArray;
  }

  public int getWinId() {
    return _winId;
  }

  public static void translateCoarray(XobjList coarrayDecl, XMPglobalDecl globalDecl,
                                      boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // FIXME delete this after implementing
    if (isLocalPragma) {
      throw new XMPexception("coarray is now allowed in a function");
    }

    // start translation
    XMPsymbolTable localXMPsymbolTable = null;
    if (isLocalPragma) {
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    }

    // find aligned array
    String coarrayName = coarrayDecl.getArg(0).getString();
    XMPalignedArray alignedArray = null;
    if (isLocalPragma) {
      alignedArray = localXMPsymbolTable.getXMPalignedArray(coarrayName);
    }
    else {
      alignedArray = globalDecl.getXMPalignedArray(coarrayName);
    }

    // FIXME??? allow an aligned array to be a coarray?
    if (alignedArray != null) {
      throw new XMPexception("an aligned array cannot be declared as a coarray");
    }

    // FIXME delete this
    System.out.println("coarray: " + coarrayName + "[" + coarrayDecl.getArg(1).toString() + "]");
  }
}
