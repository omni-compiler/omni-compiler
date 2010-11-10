/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.util.Vector;
import java.util.Iterator;

public class XMPalignedArray {
  // defined in xmp_constant.h
  public final static int NOT_ALIGNED	= 200;
  public final static int DUPLICATION	= 201;
  public final static int BLOCK		= 202;
  public final static int CYCLIC	= 203;

  private String		_name;
  private Xtype			_type;
  private int			_dim;
  private Vector<Long>		_sizeVector;
  private Vector<XMPshadow>	_shadowVector;
  private Vector<Integer>	_alignMannerVector;
  private Vector<Ident>		_accIdVector;
  private Vector<Ident>		_gtolTemp0IdVector;
  private Vector<Integer>	_alignSubscriptIndexVector;
  private Vector<Xobject>	_alignSubscriptExprVector;
  private Ident			_arrayId;
  private Ident			_descId;
  private Ident			_addrId;
  private boolean		_hasShadow;
  private boolean		_reallocChecked;
  private boolean		_realloc;
  private XMPtemplate		_alignedTemplate;

  public static int convertDistMannerToAlignManner(int distManner) throws XMPexception {
    switch (distManner) {
      case XMPtemplate.DUPLICATION:
        return DUPLICATION;
      case XMPtemplate.BLOCK:
        return BLOCK;
      case XMPtemplate.CYCLIC:
        return CYCLIC;
      default:
        throw new XMPexception("unknown dist manner");
    }
  }

  public XMPalignedArray(String name, Xtype type, int dim,
                         Vector<Long> sizeVector, Vector<Ident> accIdVector,
                         Ident arrayId, Ident descId, Ident addrId,
                         XMPtemplate alignedTemplate) {
    _name = name;
    _type = type;
    _dim = dim;
    _sizeVector = sizeVector;
    _shadowVector = new Vector<XMPshadow>(XMP.MAX_DIM);
    _alignMannerVector = new Vector<Integer>(XMP.MAX_DIM);
    _accIdVector = accIdVector;
    _gtolTemp0IdVector = new Vector<Ident>(XMP.MAX_DIM);
    _alignSubscriptIndexVector = new Vector<Integer>(XMP.MAX_DIM);
    _alignSubscriptExprVector = new Vector<Xobject>(XMP.MAX_DIM);
    for (int i = 0; i < dim; i++) {
      _shadowVector.add(new XMPshadow(XMPshadow.SHADOW_NONE, null, null));
      _alignMannerVector.add(null);
      _gtolTemp0IdVector.add(null);
      _alignSubscriptIndexVector.add(null);
      _alignSubscriptExprVector.add(null);
    }
    _arrayId = arrayId;
    _descId = descId;
    _addrId = addrId;
    _hasShadow = false;
    _reallocChecked = false;
    _alignedTemplate = alignedTemplate;
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

  public void setAlignMannerAt(int manner, int index) {
    _alignMannerVector.setElementAt(new Integer(manner), index);
  }

  public int getAlignMannerAt(int index) {
    return _alignMannerVector.get(index).intValue();
  }

  public String getAlignMannerStringAt(int index) throws XMPexception {
    switch (getAlignMannerAt(index)) {
      case NOT_ALIGNED:
        return new String("NOT_ALIGNED");
      case DUPLICATION:
        return new String("DUPLICATION");
      case BLOCK:
        return new String("BLOCK");
      case CYCLIC:
        return new String("CYCLIC");
      default:
        throw new XMPexception("unknown align manner");
    }
  }

  public Vector<Ident> getAccIdVector() {
    return _accIdVector;
  }

  public Ident getAccIdAt(int index) {
    return _accIdVector.get(index);
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

  public void setAlignSubscriptIndexAt(int alignSubscriptIndex, int alignSourceIndex) {
    _alignSubscriptIndexVector.setElementAt(new Integer(alignSubscriptIndex), alignSourceIndex);
  }

  public Integer getAlignSubscriptIndexAt(int alignSourceIndex) {
    return _alignSubscriptIndexVector.get(alignSourceIndex);
  }

  public void setAlignSubscriptExprAt(Xobject alignSubscriptExpr, int alignSourceIndex) {
    _alignSubscriptExprVector.setElementAt(alignSubscriptExpr, alignSourceIndex);
  }

  public Xobject getAlignSubscriptExprAt(int alignSourceIndex) {
    return _alignSubscriptExprVector.get(alignSourceIndex);
  }

  public Ident getArrayId() {
    return _arrayId;
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

  public XMPtemplate getAlignedTemplate() {
    return _alignedTemplate;
  }

  public boolean checkRealloc() throws XMPexception {
    if (_reallocChecked) return _realloc;

    if (_hasShadow) {
      for (int i = 0; i < _dim; i++) {
        switch (getAlignMannerAt(i)) {
          case NOT_ALIGNED:
          case DUPLICATION:
            break;
          case BLOCK:
          case CYCLIC:
            {
              XMPshadow shadow = getShadowAt(i);
              switch (shadow.getType()) {
                case XMPshadow.SHADOW_FULL:
                  break;
                case XMPshadow.SHADOW_NONE:
                case XMPshadow.SHADOW_NORMAL:
                  {
                    _reallocChecked = true;
                    _realloc = true;
                    return _realloc;
                  }
                default:
                  throw new XMPexception("unknown shadow type");
              }
            } break;
          default:
            throw new XMPexception("unknown align manner");
        }
      }

      _reallocChecked = true;
      _realloc = false;
      return _realloc;
    }
    else {
      for (int i = 0; i < _dim; i++) {
        switch (getAlignMannerAt(i)) {
          case NOT_ALIGNED:
          case DUPLICATION:
            break;
          case BLOCK:
          case CYCLIC:
            {
              _reallocChecked = true;
              _realloc = true;
              return _realloc;
            }
          default:
            throw new XMPexception("unknown align manner");
        }
      }

      _reallocChecked = true;
      _realloc = false;
      return _realloc;
    }
  }

  public boolean realloc() throws XMPexception {
    if (_reallocChecked) return _realloc;
    else                 return checkRealloc();
  }
}
