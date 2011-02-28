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
  private int			_varDim;
  private Vector<Long>		_sizeVector;
  private Ident			_varId;
  private Ident			_XMPdescId;
  private Ident			_CommDescId;

  public XMPcoarray(String name, Xtype elmtType, int varDim, Vector<Long> sizeVector,
                    Ident varId, Ident XMPdescId, Ident CommDescId) {
    _name = name;
    _elmtType = elmtType;
    _varDim = varDim;
    _sizeVector = sizeVector;
    _varId = varId;
    _XMPdescId = XMPdescId;
    _CommDescId = CommDescId;
  }

  public String getName() {
    return _name;
  }

  public Xtype getElmtType() {
    return _elmtType;
  }

  public int getVarDim() {
    return _varDim;
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
}
