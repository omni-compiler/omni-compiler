package exc.openacc;
import exc.object.*;

import java.util.*;

public class ACCvar {
  private Ident id;
  private ACCpragma atr;
  
  private ACCvar parent = null;
  
  private boolean isSpecifiedDataAttribute = false;
  
  //for data clause
  private boolean isPresent = false;
  private boolean isPresentOr = false;
  private boolean allocatesDeviceMemory = false;
  private boolean copyHostToDevice = false;
  private boolean copyDeviceToHost = false;
  private Ident deviceptr = null;
  private Ident hostDesc = null;
  
  //for reduction clause
  private ACCpragma reductionOp = null;
  
  //for parallel, kernels directive
  private boolean isFirstprivate = false;
  private boolean isPrivate = false;
  
  //for cache directive
  private boolean isCache = false;
  
  //for use_device clause
  private boolean isUse_device = false;
  
  //for subarray
  private XobjList subscriptList = null;
  
  private Subarray g_subarray = null;
  
  private Set<Subarray> subarraySet = new HashSet<Subarray>();
  
  ACCvar(Ident id, ACCpragma atr, ACCvar parent) throws ACCexception{
    this(id, null, atr, parent);
  }
  ACCvar(Ident id, XobjList subscripts, ACCpragma atr, ACCvar parent) throws ACCexception{
    this.id = id;
    this.atr = atr;
    this.parent = parent;
    ArrayRange [] arrayRangeArray;
    
    if(subscripts != null && !subscripts.isEmpty()){
      arrayRangeArray = makeArrayRangeArray(subscripts);
      g_subarray = new Subarray(arrayRangeArray, atr);
      subscriptList = subscripts;//cache の動作確認用に一時的に追加
    }else{
      //arrayRangeArray = makeArrayRangeArray(id.Type());
    }
    setAttribute(atr);

    if(parent != null && id == parent.getId()){
      //inherit parent's attribute
//      deviceptr = parent.deviceptr;
//      reductionOp = parent.reductionOp;
//      isFirstprivate = parent.isFirstprivate;
//      isPrivate = parent.isPrivate;
//      isCache = parent.isCache;
//      isUse_device = parent.isUse_device;
    }
  }
  
  public void setAttribute(ACCpragma atr) throws ACCexception{
    if(atr.isDataClause() && isSpecifiedDataAttribute){
      ACC.fatal("ACCvar: " + id.getName() + " is already specified data attribute");
    }
    
    switch(atr){
    case COPY:
      allocatesDeviceMemory = copyHostToDevice = copyDeviceToHost = true;
      break;
    case COPYIN:
      allocatesDeviceMemory = copyHostToDevice = true;
      break;
    case COPYOUT:
      allocatesDeviceMemory = copyDeviceToHost = true;
      break;
    case CREATE:
      allocatesDeviceMemory = true;
      break;
    case PRESENT:
      isPresent = true;
      break;
    case PRESENT_OR_COPY:
      isPresentOr = true;
      allocatesDeviceMemory = copyHostToDevice = copyDeviceToHost = true;
      break;
    case PRESENT_OR_COPYIN:
      isPresentOr = true;
      allocatesDeviceMemory = copyHostToDevice = true;
      break;
    case PRESENT_OR_COPYOUT:
      isPresentOr = true;
      allocatesDeviceMemory = copyDeviceToHost = true;
      break;
    case PRESENT_OR_CREATE:
      isPresentOr = true;
      allocatesDeviceMemory = true;
      break;
    case DEVICEPTR:
      deviceptr = id;
      break;
    case PRIVATE:
      isPrivate = true;
      break;
    case FIRSTPRIVATE:
      isFirstprivate = true;
      break;
    case CACHE:
      isCache = true;
      break;
    case HOST:
      copyDeviceToHost = true;
      break;
    case DEVICE:
      copyHostToDevice = true;
      break;
    case USE_DEVICE:
      isUse_device = true;
      break;
    default:
      if(atr.isReduction()){
        reductionOp = atr;
      }else{
        throw new ACCexception("var:"+id.getName()+", attribute:" + atr +" is not valid");
      } 
    }
  }
  
//  public void setAttribute(ACCpragma atr) throws ACCexception{
//    switch(atr){
//    case DEVICEPTR:
//      deviceptr = id;
//      break;
//    case USE_DEVICE:
//      isUse_device = true;
//      break;
//    default:
//      setAttribute(atr, makeArrayRangeArray(id.Type()));
//    }
//  }
//  public void setAttribute(ACCpragma atr, XobjList subscripts) throws ACCexception{
//    Subarray subarray = getSubarray(makeArrayRangeArray(subscripts));
//    if(subarray != null){
//      subarray.setAttribute(atr);
//    }
//  }
//  public void setAttribute(ACCpragma atr, ArrayRange [] arrayRangeArray) throws ACCexception{
//    Subarray subarray = getSubarray(arrayRangeArray);
//    if(subarray != null){
//      subarray.setAttribute(atr);
//    }
//  }
  
  public String getName(){
    return id.getName();
  }

  @Override
  public String toString(){
    return getName();
  }
  
  public boolean isPresent(){
    return isPresent;
  }
  public boolean isPresentOr(){
    return isPresentOr;
  }
  public boolean allocatesDeviceMemory(){
    return allocatesDeviceMemory;
  }
  public boolean copiesHtoD(){
    return copyHostToDevice;
  }
  public boolean copiesDtoH(){
    return copyDeviceToHost;
  }

  public boolean isPrivate(){
    return isPrivate;
  }
  public boolean isFirstprivate(){
    return isFirstprivate;
  } 
  public boolean isReduction(){
    return (reductionOp != null);
  }
  public boolean isCache(){
    return isCache;
  }
/*
//  public boolean isPresent(){ 
//    for(Subarray subarray : subarraySet){ //サブアレイ指定されていない部分はどうするのか？
//      if(! subarray.attribute.isPresent){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean isPresentOr(){
//    for(Subarray subarray : subarraySet){
//      if(! subarray.attribute.isPresentOr){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean isReduction(){
//    for(Subarray subarray : subarraySet){
//      if(subarray.attribute.reductionOp == null){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean isPrivate(){
//    for(Subarray subarray : subarraySet){
//      if(! subarray.attribute.isPrivate){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean isFirstprivate(){
//    for(Subarray subarray : subarraySet){
//      if(! subarray.attribute.isFirstprivate){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean isCache(){
//    for(Subarray subarray : subarraySet){
//      if(! subarray.attribute.isCache){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean allocatesDeviceMemory(){
//    for(Subarray subarray : subarraySet){
//      if(! subarray.attribute.allocatesDeviceMemory){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean copiesHtoD(){
//    for(Subarray subarray : subarraySet){
//      if(! subarray.attribute.copyHostToDevice){
//        return false;
//      }
//    }
//    return true;
//  }
//  public boolean copiesDtoH(){
//    for(Subarray subarray : subarraySet){
//      if(! subarray.attribute.copyDeviceToHost){
//        return false;
//      }
//    }
//    return true;
//  }
*/
  public Ident getId(){
    return id;
  }
  public boolean isUse_device(){
    return isUse_device;
  }
  public Ident getDevicePtr(){
    return deviceptr;
  }
  public void setDevicePtr(Ident devicePtr){
    this.deviceptr = devicePtr;
  }
  public void setHostDesc(Ident hostDesc){
    this.hostDesc = hostDesc;
  }
  public Ident getHostDesc(){
    return hostDesc;
  }
  public boolean isAllocated(){
    //return deviceptr != null;
    //return allocatesDeviceMemory();
    return (deviceptr != null ) || allocatesDeviceMemory;
  }
  public ACCpragma getReductionOperator(){
    return reductionOp;
  }
  public boolean contains(XobjList subscripts){
    return true;
  }
  private XobjList getSubscripts(Xtype type) throws ACCexception{
    if(type.isArray()){
      ArrayType arrayType = (ArrayType)type;
      long arraySize = arrayType.getArraySize();
      Xobject arraySizeObj;
      if ((arraySize == 0)){// || (arraySize == -1)) {
        throw new ACCexception("array size should be declared statically");
      }else if(arraySize==-1){
        arraySizeObj = arrayType.getArraySizeExpr();
      }else{
        arraySizeObj = Xcons.LongLongConstant(0,arraySize);
      }
      XobjList subscripts = getSubscripts(arrayType.getRef());
      subscripts.cons(makeSubscript(Xcons.IntConstant(0), arraySizeObj));
      return subscripts;
    }else{
      return Xcons.List(); //Xcons.List(Xcons.List(Xcons.IntConstant(0), Xcons.IntConstant(1)));
    }
  }
  
  private XobjList makeSubscript(Xobject lower, Xobject length){
    return Xcons.List(lower, length);
  }
//  private XobjList fixSubscript(XobjList subscripts) throws ACCexception{
//    XobjList arrayDim = getSubscripts(id.Type());
//    
//    XobjList result = Xcons.List();
//    int dim = 0;
//    for(Xobject sub : subscripts){
//      if(sub.Opcode()!=Xcode.LIST){
//        result.add(ACCutil.foldIntConstant(sub));
//      }else{
//        Xobject lower = sub.getArg(0);
//        Xobject length = sub.getArgOrNull(1);
//        lower = ACCutil.foldIntConstant(lower);
//        if(length==null){
//          length = Xcons.binaryOp(Xcode.MINUS_EXPR, arrayDim.getArg(dim).getArg(1), lower);
//        }
//        length = ACCutil.foldIntConstant(length);
//        result.add(Xcons.List(lower,length));
//      }
//      dim++;
//    }
//    return result;
//  }
  private List<ArrayRange> makeSubscripts(XobjList xSubscriptList) throws ACCexception{
    List<ArrayRange> subscriptList = new ArrayList<ArrayRange>();
    
    Xtype type = id.Type();
    for(Xobject xs : xSubscriptList){
      if(! type.isArray()){
        throw new ACCexception("too many subscripts");
      }
      ArrayType arrayType = (ArrayType)type;
      if(xs.Opcode() != Xcode.LIST){
        subscriptList.add(new ArrayRange(xs, 1));
      }else{
        Xobject lower = xs.getArg(0);
        Xobject length = xs.getArgOrNull(1);
        if(length == null){
          length = Xcons.binaryOp(Xcode.MINUS_EXPR, getArraySize(arrayType), lower);
        }
        subscriptList.add(new ArrayRange(lower, length));
      }
      type = arrayType.getRef();//getArrayElementType();
    }
    if(type.isArray()){
      throw new ACCexception("too few subscripts");
    }
    return subscriptList;
  }
  
  private ArrayRange [] makeArrayRangeArray(XobjList xSubscriptList) throws ACCexception{
    //int arrayDim = id.Type().getNumDimensions();
      
    List<ArrayRange> arrayRangeList = new ArrayList<ArrayRange>();
    //List<ArrayRange> subscriptList = new ArrayList<ArrayRange>();
    
    Xtype type = id.Type();
    //int dim = 0;
    for(Xobject xs : xSubscriptList){
	switch(type.getKind()){
	case Xtype.ARRAY:
	    ArrayType arrayType = (ArrayType)type;
	    if(xs.Opcode() != Xcode.LIST){
		//arrayRangeArray[dim] = new ArrayRange(xs, 1);
		arrayRangeList.add(new ArrayRange(xs, 1));
	    }else{
		Xobject lower = xs.getArg(0);
		Xobject length = xs.getArgOrNull(1);
		if(length == null){
		    length = Xcons.binaryOp(Xcode.MINUS_EXPR, getArraySize(arrayType), lower);
		}
		//arrayRangeArray[dim] = new ArrayRange(lower, length);
		arrayRangeList.add(new ArrayRange(lower, length));
	    }
	    type = arrayType.getRef();
	    break;
	case Xtype.POINTER:
	    if(xs.Opcode() == Xcode.LIST){
		Xobject lower = xs.getArg(0);
		Xobject length = xs.getArgOrNull(1);
		if(length != null){
		    arrayRangeList.add(new ArrayRange(lower, length));
		    break;
		}
	    }
	    throw new ACCexception("unshaped pointer");
	default:
	    throw new ACCexception("too many subscripts");   
	}

      //dim++;
    }
    if(type.isArray()){
      throw new ACCexception("too few subscripts");
    }
    
    ArrayRange [] arrayRangeArray = new ArrayRange[arrayRangeList.size()];
    for(int i = 0; i < arrayRangeList.size(); i++){
	arrayRangeArray[i] = arrayRangeList.get(i);
    }

    return arrayRangeArray;
  }
  
  //make ArrayRangeArray from id.Type()
  private ArrayRange [] makeArrayRangeArray(Xtype type){
    int arrayDim = type.getNumDimensions();
    ArrayRange [] arrayRangeArray = new ArrayRange[arrayDim];

    for(int i = 0; i < arrayDim; i++){
      if(type.isArray()){
        arrayRangeArray[i] = new ArrayRange(0, getArraySize((ArrayType)type));
        type = ((ArrayType)type).getRef();
      }
    }
    return arrayRangeArray;
  }

  private Xobject getArraySize(ArrayType arrayType){
      long arraySize = arrayType.getArraySize();
      if (arraySize <= 0){
        return arrayType.getArraySizeExpr();
      }
      if(arraySize > Integer.MAX_VALUE){
        return Xcons.LongLongConstant(0,arraySize);
      }
      else{
        return Xcons.IntConstant((int)arraySize);
      }
  }
  
  private Subarray getSubarray(ArrayRange [] arrayRangeArray){
    for(Subarray subarray : subarraySet){
      if(subarray.isSameRangeWith(arrayRangeArray)){
        return subarray;
      }
    }
    return null;
  }
  
  public Xobject getAddress() throws ACCexception{
    Xobject addrObj = null;
    Xtype varType = id.Type();
    
    switch (varType.getKind()) {
    case Xtype.BASIC:
    case Xtype.STRUCT:
    case Xtype.UNION:
      addrObj = id.getAddr();
      break;
    case Xtype.POINTER:
    {
	if(isSubarray()){
	    addrObj = id.Ref();
	}else{
	    addrObj = id.getAddr();
	}
	break;
    }
    case Xtype.ARRAY:
    {
      ArrayType arrayVarType = (ArrayType)varType;
      switch (arrayVarType.getArrayElementType().getKind()) {
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.UNION:
        break;
      default:
        throw new ACCexception("array '" + getName() + "' has a wrong data type for acc data");
      }

      addrObj = id.Ref();
      break;
    }
    default:
      throw new ACCexception("'" + getName() + "' has a wrong data type for acc data");
    }
    return addrObj;
  }
  public Xobject getSize() throws ACCexception{
    Xobject sizeObj = null;
    Xtype varType = id.Type();
    
    switch (varType.getKind()) {
    case Xtype.ARRAY:
    {
      ArrayType arrayVarType = (ArrayType)varType;
      Xtype arrayElementType = varType.getArrayElementType();
      if(isSubarray()){
        sizeObj = Xcons.binaryOp(Xcode.MUL_EXPR, g_subarray.getNumberOfElement(), Xcons.SizeOf(arrayElementType)); 
      }else{
        sizeObj = Xcons.binaryOp(Xcode.MUL_EXPR, 
            ACCutil.getArrayElmtCountObj(arrayVarType),
            Xcons.SizeOf(((ArrayType)varType).getArrayElementType()));
      }
      break;
    }
    case Xtype.POINTER:
    {
	Xtype elementType = getElementType(varType);//varType.getRef(); //TODO support pointer of pointer of int
	if(isSubarray()){
	    sizeObj = Xcons.binaryOp(Xcode.MUL_EXPR, g_subarray.getNumberOfElement(), Xcons.SizeOf(elementType));
	}else{
	    sizeObj = Xcons.SizeOf(varType);
	}
	break;
    }
    default:
      sizeObj = Xcons.SizeOf(varType);
      break;
    }
    return sizeObj;
  }
  public Xobject getNumElements() throws ACCexception{
      Xobject elementsObj = null;
      Xtype varType = id.Type();
      
      switch (varType.getKind()) {
      case Xtype.ARRAY:
      {
        ArrayType arrayType = (ArrayType)varType;
        Xtype arrayElementType = varType.getArrayElementType();
        if(isSubarray()){
          elementsObj = g_subarray.getNumberOfElement();
        }else{
          elementsObj = ACCutil.getArrayElmtCountObj(arrayType);
        }
        break;
      }
      case Xtype.POINTER:
      {
          Xtype elementType = getElementType(varType);//varType.getRef(); //TODO support pointer of pointer of int
          if(isSubarray()){
              elementsObj = g_subarray.getNumberOfElement();
          }else{
              elementsObj = Xcons.IntConstant(1);
          }
          break;
      }
      default:
        elementsObj = Xcons.IntConstant(1);//Xcons.SizeOf(varType);
        break;
      }
      return elementsObj;
    }
  public Xobject getOffset() throws ACCexception{
    Xobject offsetObj = null;
    Xtype varType = id.Type();
    
    switch (varType.getKind()) {
    case Xtype.ARRAY:
    {
      Xtype arrayElementType = varType.getArrayElementType();
      if(isSubarray()){
        offsetObj = Xcons.binaryOp(Xcode.MUL_EXPR, g_subarray.getOffsetCount(), Xcons.SizeOf(arrayElementType)); 
      }else{
        offsetObj = Xcons.IntConstant(0);
      }
      break;
    }
    default:
      offsetObj = Xcons.IntConstant(0);
      break;
    }
    return offsetObj;
  }
  
  /////////////////
  
  public boolean conllidesWith(XobjList subscripts){
    int dim = Math.max(this.subscriptList.Nargs(), subscripts.Nargs());
    for(int i=0;i<dim;i++){
      Xobject s1 = this.subscriptList.getArg(i);
      Xobject s2 = subscripts.getArg(i);
      long low1=0,len1=1,low2=0,len2=1;
      try{
        if(s1.Opcode() != Xcode.LIST){ //s1 is single
          low1 = toLong(s1);
          //len1 = 1;
        }else{
          low1 = toLong(s1.getArg(0));
          len1 = toLong(s1.getArgOrNull(1));
        }
        if(s2.Opcode() != Xcode.LIST){ //s2 is single
          low2 = toLong(s2);
          //len1 = 1;
        }else{
          low2 = toLong(s2.getArg(0));
          len2 = toLong(s2.getArgOrNull(1));
        }
      }catch(ACCexception e){
        return false;
      }
      if(hasIntersect(low1, len1, low2, len2)) return true;
    }
    return false;
  }
  private long toLong(Xobject x) throws ACCexception{
    if(x == null){
      throw new ACCexception("null");
    }
    if(x.isIntConstant()){
      return (long)x.getInt();
    }else if(x.Opcode()== Xcode.LONGLONG_CONSTANT){
      return (long)x.getLong();
    }throw new ACCexception("not constant");

  }
  private boolean hasIntersect(long low1, long len1, long low2, long len2){
    if(low1 < low2){
      return (low1+len1 > low2);
    }else{
      return (low2+len2 > low1);
    }
  }
  public XobjList getSubscripts(){
    return subscriptList;
  }
  
  boolean isSubarray(){
    return g_subarray != null;
  }
  
  Xtype getElementType(Xtype t)
  {
      if(t.isArray()){
          return ((ArrayType)t).getArrayElementType();
      }else if(t.isPointer()){
          return getElementType(t.getRef());
      }else if(t.isBasic()){
          return t;
      }else{
          return null;
      }
  }

  class Subarray{
    private Attribute attribute;
    private ArrayRange [] arrayRangeArray;
    Subarray(ArrayRange [] arrayRangeArray){
      this.arrayRangeArray = arrayRangeArray;
    }
    Subarray(ArrayRange [] arrayRangeArray, ACCpragma atr) throws ACCexception{
      this.arrayRangeArray = arrayRangeArray;
      attribute = new Attribute(atr);
    }
    void setAttribute(ACCpragma atrPragma) throws ACCexception{
      attribute.setAttribute(atrPragma);
    }
    boolean hasIntersection(ArrayRange [] arrayRangeArray){
      int dim = this.arrayRangeArray.length;
      if(arrayRangeArray.length != dim){
        return false;
      }
      for(int i = 0; i < dim; i++){
        if((this.arrayRangeArray)[i].hasIntersection(arrayRangeArray[i])){
          return true;
        }
      }
      return false;
    }
    boolean isSameRangeWith(ArrayRange [] arrayRangeArray){
      int dim = this.arrayRangeArray.length;
      if(arrayRangeArray.length != dim){
        return false;
      }
      for(int i = 0; i < dim; i++){
        if((this.arrayRangeArray)[i].isSameWith(arrayRangeArray[i])){
          return true;
        }
      }
      return false;
    }
    Xobject getNumberOfElement(){
      int dim = this.arrayRangeArray.length;
      Xobject numberOfElement = Xcons.IntConstant(1);
      for(int i = 0; i < dim; i++){
        numberOfElement = Xcons.binaryOp(Xcode.MUL_EXPR, numberOfElement, arrayRangeArray[i].getLengthObj());
      }
      return numberOfElement;
    }
    Xobject getOffsetCount(){
      int dim = this.arrayRangeArray.length;
      if(dim == 0){
        return Xcons.IntConstant(0);
      }
      Xobject numberOfLowerElement = Xcons.IntConstant(1);
      for(int i = 1; i < dim; i++){
        numberOfLowerElement = Xcons.binaryOp(Xcode.MUL_EXPR, numberOfLowerElement, arrayRangeArray[i].getLengthObj());
      }
      return Xcons.binaryOp(Xcode.MUL_EXPR, arrayRangeArray[0].getLowerObj(), numberOfLowerElement);
    }
  }
  
  class Attribute{
    //for data clause
    private boolean isPresent = false;
    private boolean isPresentOr = false;
    private boolean allocatesDeviceMemory = false; //it may be unnecessary
    private boolean copyHostToDevice = false;
    private boolean copyDeviceToHost = false;
    
    //for reduction clause
    private ACCpragma reductionOp = null;
    
    //for parallel, kernels directive
    private boolean isFirstprivate = false;
    private boolean isPrivate = false;
    
    //for cache directive
    private boolean isCache = false;

    private boolean isSpecifiedDataAttribute = false;

    public Attribute(ACCpragma atr) throws ACCexception{
      setAttribute(atr);
    }
    
    public void setAttribute(ACCpragma atr) throws ACCexception{
      if(atr.isDataClause() && isSpecifiedDataAttribute){
        throw new ACCexception("data attribute is already specified");
      }
      if(atr.isDataClause()){
        isSpecifiedDataAttribute = true;  
      }
      
      switch(atr){
      case COPY:
        allocatesDeviceMemory = copyHostToDevice = copyDeviceToHost = true;
        break;
      case COPYIN:
        allocatesDeviceMemory = copyHostToDevice = true;
        break;
      case COPYOUT:
        allocatesDeviceMemory = copyDeviceToHost = true;
        break;
      case CREATE:
        allocatesDeviceMemory = true;
        break;
      case PRESENT:
        isPresent = true;
        break;
      case PRESENT_OR_COPY:
        isPresentOr = true;
        allocatesDeviceMemory = copyHostToDevice = copyDeviceToHost = true;
        break;
      case PRESENT_OR_COPYIN:
        isPresentOr = true;
        allocatesDeviceMemory = copyHostToDevice = true;
        break;
      case PRESENT_OR_COPYOUT:
        isPresentOr = true;
        allocatesDeviceMemory = copyDeviceToHost = true;
        break;
      case PRESENT_OR_CREATE:
        isPresentOr = true;
        allocatesDeviceMemory = true;
        break;
      case PRIVATE:
        isPrivate = true;
        break;
      case FIRSTPRIVATE:
        isFirstprivate = true;
        break;
      case CACHE:
        isCache = true;
        break;
      case HOST:
        copyDeviceToHost = true;
        break;
      case DEVICE:
        copyHostToDevice = true;
        break;
      default:
        if(atr.isReduction()){
          reductionOp = atr;
        }else{
          throw new ACCexception("invalid attribute");
        } 
      }
    }
  }
  
  class ArrayRange{
    private long longLower;
    private long longLength;
    private Xobject xLower = null, xLength = null;
    
    ArrayRange(Xobject lower, Xobject length){
      setLower(lower);
      setLength(length);
    }
    ArrayRange(Xobject lower, int length){
      setLower(lower);
      longLength = (long)length;
    }
    ArrayRange(int lower, Xobject length){
      longLower = (long)lower;
      setLength(length);
    }
    ArrayRange(int lower, int length){
      longLower = (long)lower;
      longLength = (long)length;
    }
    
    private void setLower(Xobject lower){
      try{
        longLower = toLong(lower);
      }catch(Exception e){
        xLower = lower;
      }
    }
    private void setLength(Xobject length){
      try{
        longLength = toLong(length);
      }catch(Exception e){
        xLength = length;
      }
    }
    
    @Override
    public String toString(){
      String strLower, strLength;
      strLower = isLowerConstant()? Long.toString(longLower) : xLower.toString();
      strLength = isLengthConstant()? Long.toString(longLength) : xLength.toString();
      return "[" + strLower + ":" + strLength + "]"; 
    }
    
    private long toLong(Xobject x) throws Exception{
      if(x.isIntConstant()){
        return x.getInt();
      }
      if(x.Opcode() == Xcode.LONGLONG_CONSTANT){ 
        return x.getLong();
      }
      throw new Exception();
    }
    
    public boolean isLowerConstant(){
      return xLower == null;
    }
    public boolean isLengthConstant(){
      return xLength == null;
    }
    
    public boolean hasIntersection(ArrayRange s){
      if(!this.isLowerConstant() || !s.isLowerConstant()){
        return false;
      }
      if(this.longLower > s.longLower){
        return s.hasIntersection(this);
      }
      if(this.isLengthConstant()){  //low1+len1 > low2
        return (this.longLower + this.longLength > s.longLower);
      }
      return false;
    }
    
    public boolean isSingle(){
      return (longLength == 1);
    }
    public boolean isSameWith(ArrayRange ar){
      if(isLowerConstant() && isLengthConstant()){
        return longLower == ar.longLower && longLength == ar.longLength;
      }
      return false;
    }
    public Xobject getLowerObj(){
      if(isLowerConstant()){
        return Xcons.LongLongConstant(0, longLower);
      }else{
        return xLower;
      }
    }
    public Xobject getLengthObj(){
      if(isLengthConstant()){
        return Xcons.LongLongConstant(0, longLength);
      }else{
        return xLength;
      }
    }
  }

}
