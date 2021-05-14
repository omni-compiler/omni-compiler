package exc.object;

import java.util.List;
import java.util.ArrayList;

class ASTnode {
      
  Xcode code; // NULL for leaf
  Xobject x; // arg for leaf
  List<ASTnode> child;

  int endIndex;
  int parrenLevel;
  int argPos;

  // code
  // 0 : value
  // 100 : +
  // 200 : -
  // 300 : *
  // 400 : /
  // 500 : f
  // 600 : list
  // 700 : array
  
  ASTnode(Xcode code, Xobject x, List<ASTnode> child){
    this.code = code;
    this.child = child;
    this.x = x;
    this.endIndex = -1;
    this.parrenLevel = -1;
    this.argPos = -1;
  }

  ASTnode(String format, Object... args)
  {
    ASTnode root = parse(format, 0, 0, 0, 0, args);
    this.code = root.code;
    this.child = root.child;
    this.x = root.x;
    this.endIndex = root.endIndex;
    this.parrenLevel = root.parrenLevel;
    this.argPos = root.argPos;
  }

  public int getEndIndex(){
    return this.endIndex;
  }

  public void setEndIndex(int endIndex){
    this.endIndex = endIndex;
  }

  public int getParrenLevel(){
    return this.parrenLevel;
  }

  public void setParrenLevel(int parrenLevel){
    this.parrenLevel = parrenLevel;
  }

  public int getArgPos(){
    return this.argPos;
  }

  public void setArgPos(int argPos){
    this.argPos = argPos;
  }

  Xobject toXobject(){

    switch (code){

    case PLUS_EXPR:
    case MINUS_EXPR:
    case MUL_EXPR:
    case DIV_EXPR:
      return Xcons.binaryOp(code, child.get(0).toXobject(), child.get(1).toXobject());

    case FUNCTION_CALL:
      return Xcons.functionCall(x, child.get(0).toXobject());

    case F_ARRAY_REF:
      return Xcons.FarrayRef(x, child.get(0).toXobject());
      
    case F_VAR_REF:
    case INT_CONSTANT:
      return x;

    case LIST: {
      XobjList args = Xcons.List();
      for (ASTnode arg : child){
	args.add(arg.toXobject());
      }
      return args;
    }
      
    default:
      System.out.println("Syntax error in toXobject.");
      System.exit(1);
    }

    return null;
    
  }

  @Override
  public String toString()
  {
    return this.toXobject().toString();
    // if (code == 0){
    //   return String.valueOf(x);
    // }
    // else if (code == 100){
    //   return "(" + this.child.get(0).toString() + " + " + this.child.get(1).toString() + ")";
    // }
    // else if (code == 200){
    //   return "(" + this.child.get(0).toString() + " - " + this.child.get(1).toString() + ")";
    // }
    // else if (code == 300){
    //   return this.child.get(0).toString() + " * " + this.child.get(1).toString();
    // }
    // else if (code == 400){
    //   return this.child.get(0).toString() + " / " + this.child.get(1).toString();
    // }
    // else if (code == 500 || code == 700){
    //   return x.toString() + "(" + this.child.get(0).toString() + ")";
    // }
    // else if (code == 600){
    //   String subs = "";
    //   if (this.child.size() > 0){
    // 	subs += this.child.get(0);
    //   }
    //   for (int i = 1; i < this.child.size(); i++){
    // 	subs += ", " + this.child.get(i);
    //   }
    //   return  subs;
    // }
    // else {
    //   return "error";
    // }
  }

  private static ASTnode parseSubs(String format, int start, int curr_argPos, int curr_prior, int curr_parren, Object... args)
  {

    //System.out.println("parseSubs !");
    
    List<ASTnode> subs = new ArrayList<>();

    int root_parren = curr_parren;

    int i = start;;

    assert format.charAt(i) == '(':
    i++;
    curr_parren++;
	
    while (i < format.length()){
      ASTnode sub = parse(format, ++i, curr_argPos, 0, curr_parren, args);
      subs.add(sub);
      i = sub.getEndIndex();
      curr_parren = sub.getParrenLevel();
      curr_argPos = sub.getArgPos();
      if (curr_parren == root_parren) break;
    }

    ASTnode node = new ASTnode(Xcode.LIST, null, subs);
    node.setEndIndex(i);
    node.setParrenLevel(curr_parren);
    node.setArgPos(curr_argPos);
    return node;

  }
  
  private static ASTnode parse(String format, int start, int curr_argPos, int curr_prior, int curr_parren, Object... args)
  {

    //System.out.println(format.substring(start));
    
    ASTnode root = null;
    int root_parren = curr_parren;

    int i = start;;
  LOOP: while (i < format.length()){

      char c = format.charAt(i);

      //System.out.println(" curr_pos = " + curr_pos + " at " + c);

      switch (c){

      case ' ':
	i++;
	break;
	
      case '(':
	i++;
	curr_parren++;
	break;
	
      case ')':
	//System.out.println(") " + curr_parren);
	i++;
	curr_parren--;
	if (curr_parren < 0){
	  System.out.println("Syntax error.");
	  System.exit(1);
	}
	break LOOP;
	
      case '+':
      case '-':
	{

	  if (curr_prior > 0 && curr_parren == root_parren){
	    //System.out.println(root);
	    break LOOP;
	  }

	  Xcode code = (c == '+') ? Xcode.PLUS_EXPR : Xcode.MINUS_EXPR;
	  int prior = (c == '+') ? 0 : 1;
	  
	  //System.out.println("Parsed: +");

	  List<ASTnode> child = new ArrayList<>();
	  child.add(root);

	  //System.out.println("before " + i);
	  //ASTnode right = parse(format, ++i, curr_argPos, 0, curr_parren, args);
	  ASTnode right = parse(format, ++i, curr_argPos, prior, curr_parren, args);
	  child.add(right);
	  i = right.getEndIndex();
	  curr_parren = right.getParrenLevel();
	  curr_argPos = right.getArgPos();
	  //System.out.println("after " + i);
	    
	  root = new ASTnode(code, null, child);
	  curr_prior = 0;
	  break;
	}

      case '*':
      case '/':

	{
	  Xcode code = (c == '*') ? Xcode.MUL_EXPR : Xcode.DIV_EXPR;
	  
	  //System.out.println("Parsed: *");

	  List<ASTnode> child = new ArrayList<>();
	  child.add(root);

	  //System.out.println("before " + i);
	  ASTnode right = parse(format, ++i, curr_argPos, 2, curr_parren, args);
	  child.add(right);
	  i = right.getEndIndex();
	  curr_parren = right.getParrenLevel();
	  curr_argPos = right.getArgPos();
	  //System.out.println("after " + i);
	    
	  root = new ASTnode(code, null, child);
	  curr_prior = 0;
	  break;
	}
	
      case '%':

	i++;
	c = format.charAt(i);

	if (c == 'f'){ // function
	  List<ASTnode> child = new ArrayList<>();
	  Xobject func = (Xobject)args[curr_argPos++];
	  ASTnode subs = parseSubs(format, ++i, curr_argPos, 1, curr_parren, args);
	  child.add(subs);
	  i = subs.getEndIndex();
	  curr_parren = subs.getParrenLevel();
	  curr_argPos = subs.getArgPos();
	  root = new ASTnode(Xcode.FUNCTION_CALL, func, child);
	}
	else if (c == 'a'){ // array
	  List<ASTnode> child = new ArrayList<>();
	  Xobject array = (Xobject)args[curr_argPos++];
	  ASTnode subs = parseSubs(format, ++i, curr_argPos, 1, curr_parren, args);
	  child.add(subs);
	  i = subs.getEndIndex();
	  curr_parren = subs.getParrenLevel();
	  curr_argPos = subs.getArgPos();
	  root = new ASTnode(Xcode.F_ARRAY_REF, array, child);
	}
	else if (c == 'i'){ // integer constant
	  root = new ASTnode(Xcode.INT_CONSTANT,
			     Xcons.IntConstant(((Integer)args[curr_argPos++]).intValue()), null);
	  i++;
	}
	else { // other expression
	  //System.out.println("Parsed: " + ((Integer)args[curr_argPos]).intValue());
	  root = new ASTnode(Xcode.F_VAR_REF, (Xobject)args[curr_argPos++], null);
	  i++;
	}

	break;
	
      case '0': case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
	{
	  String val = String.valueOf(c);

	  //System.out.println("Integer: " + c);
	  
	  i++;
	  while (i < format.length()){
	    char cc = format.charAt(i);
	    if ('0' <= cc && cc <= '9'){
	      val += cc;
	      i++;
	    }
	    else break;
	  }

	  root = new ASTnode(Xcode.INT_CONSTANT, Xcons.IntConstant(Integer.valueOf(val)), null);
	  break;
	}

      case ',': // only in a list
	break LOOP;

      default:
	System.out.println("Syntax error in parse (" + c + " )");
	System.exit(1);
      }

    }

    // if (curr_parren != root_parren){
    //   System.out.println("Syntax error.");
    //   System.exit(1);
    // }

    //System.out.println(root + " " + i);
    root.setEndIndex(i);
    root.setParrenLevel(curr_parren);
    root.setArgPos(curr_argPos);
    return root;
      
  }
      
}
  
