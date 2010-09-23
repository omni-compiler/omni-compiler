package exc.xcalablemp;

import exc.object.*;
import java.util.Vector;

public class XMPtemplate extends XMPobject {
  // defined in xmp_constant.h
  public final static int DUPLICATION	= 100;
  public final static int BLOCK		= 101;
  public final static int CYCLIC	= 102;

  private boolean		_isFixed;
  private boolean		_isDistributed;
  private XMPnodes		_ontoNodes;
  private Vector<XobjInt>	_ontoNodesIndexVector;
  private Vector<Integer>	_distMannerVector;
  private Vector<Xobject>	_lowerVector;

  public XMPtemplate(String name, int dim, Ident descId) {
    super(XMPobject.TEMPLATE, name, dim, descId);

    _isFixed = false;
    _isDistributed = false;
    _ontoNodes = null;
    _ontoNodesIndexVector = new Vector<XobjInt>();
    _distMannerVector = new Vector<Integer>();

    for (int i = 0; i < dim; i++) {
      _ontoNodesIndexVector.add(null);
      _distMannerVector.add(new Integer(DUPLICATION));
    }

    _lowerVector = new Vector<Xobject>();
  }

  public void setIsFixed() {
    _isFixed = true;
  }

  public boolean isFixed() {
    return _isFixed;
  }

  public void setIsDistributed() {
    _isDistributed = true;
  }

  public boolean isDistributed() {
    return _isDistributed;
  }

  public void setOntoNodes(XMPnodes nodes) {
    _ontoNodes = nodes;
  }

  public XMPnodes getOntoNodes() {
    return _ontoNodes;
  }

  public void setOntoNodesIndexAt(int nodesDimIdx, int templateDimIdx) {
    _ontoNodesIndexVector.setElementAt(Xcons.IntConstant(nodesDimIdx), templateDimIdx);
  }

  public XobjInt getOntoNodesIndexAt(int index) {
    return _ontoNodesIndexVector.get(index);
  }

  public void setDistMannerAt(int manner, int index) {
    _distMannerVector.setElementAt(new Integer(manner), index);
  }

  public int getDistMannerAt(int index) {
    return _distMannerVector.get(index).intValue();
  }

  public String getDistMannerStringAt(int index) throws XMPexception {
    switch (getDistMannerAt(index)) {
      case DUPLICATION:
        return new String("DUPLICATION");
      case BLOCK:
        return new String("BLOCK");
      case CYCLIC:
        return new String("CYCLIC");
      default:
        throw new XMPexception("unknown distribute manner");
    }
  }

  public void addLower(Xobject lower) {
    _lowerVector.add(lower);
  }

  public Xobject getLowerAt(int index) {
    return _lowerVector.get(index);
  }
}
