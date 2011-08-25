/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.util.Vector;

public class XMPobject {
  public final static int NODES		= 100;
  public final static int TEMPLATE	= 101;

  private int			_kind;
  private String		_name;
  private int			_dim;
  private Ident			_descId;
  private Vector<Xobject>	_lowerVector;
  private Vector<Xobject>	_upperVector;

  public XMPobject(int kind, String name, int dim, Ident descId) {
    _kind = kind;
    _name = name;
    _dim = dim;
    _descId = descId;
    _lowerVector = new Vector<Xobject>();
    _upperVector = new Vector<Xobject>();
  }

  public int getKind() {
    return _kind;
  }

  public String getName() {
    return _name;
  }

  public int getDim() {
    return _dim;
  }

  public Ident getDescId() {
    return _descId;
  }

  public void addLower(Xobject lower) {
    _lowerVector.add(lower);
  }

  public Xobject getLowerAt(int index) {
    return _lowerVector.get(index);
  }

  public void addUpper(Xobject upper) {
    _upperVector.add(upper);
  }

  public Xobject getUpperAt(int index) {
    return _upperVector.get(index);
  }

  public boolean checkInheritExec() {
    throw new UnsupportedOperationException();
  }
}
