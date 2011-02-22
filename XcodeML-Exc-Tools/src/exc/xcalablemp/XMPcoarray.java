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
  private Xtype			_elmtType;
  private int			_dim;
  private Vector<Integer>	_sizeVector;
  private Ident			_varId;
  private Ident			_XMPdescId;
  private Ident			_CommDescId;
  private boolean		_isArray;

  public XMPcoarray(String name, Xtype elmtType, int dim, Vector<Integer> sizeVector,
                    Ident varId, Ident XMPdescId, Ident CommDescId, boolean isArray) {
    _name = name;
    _elmtType = elmtType;
    _dim = dim;
    _sizeVector = sizeVector;
    _varId = varId;
    _XMPdescId = XMPdescId;
    _CommDescId = CommDescId;
    _isArray = isArray;
  }

  public String getName() {
    return _name;
  }

  public Xtype getElmtType() {
    return _elmtType;
  }

  public int getDim() {
    return _dim;
  }

  public int getSizeAt(int index) {
    return _sizeVector.get(index).intValue();
  }

  public Ident getVarId() {
    return _varId;
  }

  public Ident getXMPdescId() {
    return _XMPdescId;
  }

  public Ident getCommDescId() {
    return _CommDescId;
  }

  public boolean isArray() {
    return _isArray;
  }
}
