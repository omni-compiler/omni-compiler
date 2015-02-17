package exc.xcalablemp;

import exc.object.Ident;

public class XACClayoutedArray {
  private XACClayout _deviceLayout;
  private XMPalignedArray _alignedArray = null;
  private Ident _descId;
  private Ident _varId = null;
  
  public XACClayoutedArray(Ident descId, XMPalignedArray alignedArray, XACClayout layout)
  {
    _alignedArray = alignedArray;
    _deviceLayout = layout;
    _descId = descId;//alignedArray.getDescId();
  }
  
  public XACClayoutedArray(Ident descId, Ident varId, XACClayout layout)
  {
    _varId = varId;
    _deviceLayout = layout;
    _descId = descId;
  }
  
  public String getName(){
    if(_alignedArray != null) return _alignedArray.getName();
    else return _varId.getName();
  }
  public void setLayout(XACClayout layout){
    _deviceLayout = layout;
  }
  public XACClayout getLayout(){
    return _deviceLayout;
  }
  public Ident getDescId(){
    return _descId;
  }
}
