/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
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
}
