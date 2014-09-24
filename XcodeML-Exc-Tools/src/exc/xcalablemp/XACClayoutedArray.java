package exc.xcalablemp;

import exc.object.Ident;

public class XACClayoutedArray {
  private XACClayout _deviceLayout;
  private XMPalignedArray _alignedArray;
  private Ident _descId;
  
  public XACClayoutedArray(Ident descId, XMPalignedArray alignedArray, XACClayout layout)
  {
    _alignedArray = alignedArray;
    _deviceLayout = layout;
    _descId = descId;//alignedArray.getDescId();
  }
  
  public String getName(){
    return _alignedArray.getName();
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
