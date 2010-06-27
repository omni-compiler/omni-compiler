package exc.xcalablemp;

import exc.object.*;
import java.util.Vector;
import java.util.Iterator;

public class XMPalignedArray {
  public final static int NO_ALIGN	= 10;
  public final static int SIMPLE_ALIGN	= 11;

  private LineNo		_lineNo;
  private String		_name;
  private Xtype			_type;
  private int			_dim;
  private Vector<Long>		_sizeVector;
  private Vector<XMPshadow>	_shadowVector;
  private Vector<Integer>	_distMannerVector;
  private Vector<Ident>		_gtolAccIdVector;
  private Vector<Ident>		_gtolTemp0IdVector;
  private Ident			_descId;
  private Ident			_addrId;
  private boolean		_hasShadow;
  private boolean		_realloc;

  public XMPalignedArray(LineNo lineNo, String name, Xtype type, int dim,
                         Vector<Long> sizeVector, Vector<Ident> gtolAccIdVector,
                         Ident descId, Ident addrId) {
    _lineNo = lineNo;
    _name = name;
    _type = type;
    _dim = dim;
    _sizeVector = sizeVector;
    _shadowVector = new Vector<XMPshadow>(XMP.MAX_DIM);
    _distMannerVector = new Vector<Integer>(XMP.MAX_DIM);
    _gtolAccIdVector = gtolAccIdVector;
    _gtolTemp0IdVector = new Vector<Ident>(XMP.MAX_DIM);
    for (int i = 0; i < dim; i++) {
      _shadowVector.add(new XMPshadow(XMPshadow.SHADOW_NONE, null, null));
      _distMannerVector.add(new Integer(NO_ALIGN));
      _gtolTemp0IdVector.add(null);
    }
    _descId = descId;
    _addrId = addrId;
    _hasShadow = false;
    _realloc = false;
  }

  public LineNo getLineNo() {
    return _lineNo;
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

  public void setDistMannerAt(int manner, int index) {
    _distMannerVector.setElementAt(new Integer(manner), index);
  }

  public int getDistMannerAt(int index) {
    return _distMannerVector.get(index).intValue();
  }

  public String getDistMannerStringAt(int index) {
    switch (getDistMannerAt(index)) {
      case NO_ALIGN:
        XMP.fatal("exception in exc.xcalablemp.XMPalignedArray.getDistMannerStringAt(), not distributed");
        return null; // XXX not reach here
      case XMPtemplate.DUPLICATION:
        return new String("DUPLICATION");
      case XMPtemplate.BLOCK:
        return new String("BLOCK");
      case XMPtemplate.CYCLIC:
        return new String("CYCLIC");
      default:
        XMP.fatal("exception in exc.xcalablemp.XMPalignedArray.getDistMannerStringAt(), unknown distribution manner");
        return null; // XXX not reach here
    }
  }

  public Vector<Ident> getGtolAccIdVector() {
    return _gtolAccIdVector;
  }

  public Ident getGtolAccIdAt(int index) {
    return _gtolAccIdVector.get(index);
  }

  // temp0 is
  // block distribution:	parallel/serial lower	| _XCALABLEMP_gtol_lower_<array_name>_<array_dim>
  // cyclic distribution:	nodes size		| _XCALABLEMP_gtol_cycle_<array_name>_<array_dim>
  public void setGtolTemp0IdAt(Ident temp0Id, int index) {
    _gtolTemp0IdVector.setElementAt(temp0Id, index);
  }

  public Ident getGtolTemp0IdAt(int index) {
    return _gtolTemp0IdVector.get(index);
  }

  public Ident getDescId() {
    return _descId;
  }

  public Ident getAddrId() {
    return _addrId;
  }

  public void setHasShadow() {
    _hasShadow = true;
  }

  public boolean hasShadow() {
    return _hasShadow;
  }

  public void setShadowAt(XMPshadow shadow, int index) {
    _shadowVector.setElementAt(shadow, index);
  }

  public XMPshadow getShadowAt(int index) {
    return _shadowVector.get(index);
  }

  public boolean checkRealloc() {
    if (_hasShadow) {
      for (int i = 0; i < _dim; i++) {
        int distManner = getDistMannerAt(i);
        if (distManner != XMPtemplate.DUPLICATION) {
          XMPshadow shadow = getShadowAt(i);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_FULL:
              break;
            case XMPshadow.SHADOW_NONE:
            case XMPshadow.SHADOW_NORMAL:
              _realloc = true;
              return true;
            default:
              XMP.fatal("unknown shadow type");
          }
        }
      }

      _realloc = false;
      return false;
    }
    else {
      _realloc = true;
      return true;
    }
  }

  public boolean realloc() {
    return _realloc;
  }
}
