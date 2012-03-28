/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.Vector;

public class XMPcoarray {
  public final static int ACC_BIT_XOR	= 1;

  private String		_name;
  private Xtype			_elmtType;
  private int			_varDim;
  private Vector<Long>		_sizeVector;
  private Xobject		_varAddr;
  private Ident			_varId;
  private Ident			_descId;

  public XMPcoarray(String name, Xtype elmtType, int varDim, Vector<Long> sizeVector,
                    Xobject varAddr, Ident varId, Ident descId) {
    _name = name;
    _elmtType = elmtType;
    _varDim = varDim;
    _sizeVector = sizeVector;
    _varAddr = varAddr;
    _varId = varId;
    _descId = descId;
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

  public Xobject getVarAddr() {
    return _varAddr;
  }

  public Ident getVarId() {
    return _varId;
  }

  public Ident getDescId() {
    return _descId;
  }
}
