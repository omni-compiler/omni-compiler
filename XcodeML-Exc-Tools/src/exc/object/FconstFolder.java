/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */

package exc.object;
import exc.block.Block;
import java.util.*;
import java.math.BigInteger;

public class FconstFolder
{
  public boolean verbose = false;

  private Block block;
  private Xobject expr;

  // case: FUNCTION CALL
  private String fname;
  private XobjList argList;
  private int nargs;

  // case: binary or unary operation
  private XobjList oprds;

  // status of _getArgAsInt(), _getIntConstArg(), _getArg()
  private boolean failed = false;


  /*************************************************\
   * CONSTRUCTOR
  \*************************************************/
  public FconstFolder(Xobject expr)
  {
    this(expr, null);
  }

  public FconstFolder(Xobject expr, Block block)
  {
    this.expr = expr;
    this.block = block;
  }


  /*************************************************\
   * EXECUTOR    
  \*************************************************/
  public Xobject run()
  {
    if (verbose)
      _info("Valur of " + expr);

    Xobject result = evalAsInitializationExpr();

    if (verbose)
      _info("      is " + result);

    return result;
  }


  public Xobject evalAsInitializationExpr()
  {
    Xobject result;

    /* constant (no evaluation)
     */ 
    result = evalAsScalarIntConstant();
    if (result != null) 
      return result;

    /* case FUNCTION CALL
     */
    if (expr.Opcode() == Xcode.FUNCTION_CALL) {

      fname = expr.left().getName().toLowerCase();

      argList = (XobjList)expr.right();
      nargs = 0;
      if (argList != null)
        nargs = argList.Nargs();

      /* inquire functions (top-down evaluation)
       */
      result = evalAsInquireIntrinsicFunction();
      if (result != null) 
        return result;

      /* elemental intrinsic functions (bottom-up evaluation)
       */
      result = evalAsElementalIntrinsicFunction();
      if (result != null) 
        return result;
    }

    /* evaluate all arguments
     */
    else {
      oprds = (XobjList)expr;
      result = null;
      Xobject arg1, arg2;

      switch (oprds.Nargs()) {
      case 1:
        arg1 = oprds.getArg(0).cfold(block);     // do constant folding
        if(arg1.isIntConstant())
          result = evalAsIntrinsicUnaryOp(arg1.getInt());
        break;

      case 2:
        arg1 = oprds.getArg(0).cfold(block);     // do constant folding
        arg2 = oprds.getArg(1).cfold(block);     // do constant folding
        if(arg1.isIntConstant() && arg2.isIntConstant())
          result = evalAsIntrinsicBinaryOp(arg1.getInt(), arg2.getInt());
        break;

      default:
        break;
      }
      if (result != null)
        return result;
    }

    return null;
  }


  /*************************************************\
   * constant
  \*************************************************/
  private Xobject evalAsScalarIntConstant()
  {
    if (expr.canGetInt())
      return expr;

    return null;
  }
      

  /*************************************************\
   * inquire intrinsic functions
  \*************************************************/
  private Xobject evalAsInquireIntrinsicFunction()
  {
    if (fname == "lbound") {
      if (nargs == 1)
        return _evalIntrinsic_lboundWithoutDim();
      else if (nargs == 2)
        return _evalIntrinsic_lboundWithDim();
    }
    else if (fname == "ubound") {
      if (nargs == 1)
        return _evalIntrinsic_uboundWithoutDim();
      else if (nargs == 2)
        return _evalIntrinsic_uboundWithDim();
    }
    else if (fname == "shape") {
      if (nargs == 1)
        return _evalIntrinsic_shape();
    }
    else if (fname == "size") {
      if (nargs == 1)
        return _evalIntrinsic_sizeWithoutDim();
      else if (nargs == 2)
        return _evalIntrinsic_sizeWithDim();
    }
    else if (fname == "lcobound")
      return _evalIntrinsic_lcobound();
    else if (fname == "ucobound")
      return _evalIntrinsic_ucobound();

    else if (fname == "bit_size")
      ;
    else if (fname == "len")
      ;

    else if (fname == "kind")
      return _evalIntrinsic_kind();

    else if (fname == "digits")
      ;
    else if (fname == "epsilon")
      ;
    else if (fname == "huge")
      ;
    else if (fname == "maxexponent")
      ;
    else if (fname == "minexponent")
      ;
    else if (fname == "precision")
      ;
    else if (fname == "radix")
      ;
    else if (fname == "range")
      ;
    else if (fname == "tiny")
      ;
    else if (fname == "repeat")
      ;
    else if (fname == "reshape")
      ;
    else if (fname == "selected_int_kind")
      ;
    else if (fname == "selected_real_kind")
      ;
    else if (fname == "transfer")
      ;
    else if (fname == "trim")
      ;
    else if (fname == "len" && nargs == 1)
      ;

    return null;
  }


  private Xobject _evalIntrinsic_lboundWithoutDim()
  {
    Xobject arg_array = _getArg("array", 0);
    if (failed)
      return null;

    // not supported yet
    _error("cannot evaluate intrinsic function \'lbound\' without \'dim\'");
    return null;
  }

  private Xobject _evalIntrinsic_lboundWithDim()
  {
    Xobject arg_array = _getArg("array", 0);
    int dim = _getArgAsInt("dim", 1);
    if (failed)
      return null;

    return arg_array.lbound(dim-1, block);
  }


  private Xobject _evalIntrinsic_uboundWithoutDim()
  {
    Xobject arg_array = _getArg("array", 0);
    if (failed)
      return null;

    // not supported yet
    _error("cannot evaluate intrinsic function \'lbound\' without \'dim\'");
    return null;
  }

  private Xobject _evalIntrinsic_uboundWithDim()
  {
    Xobject arg_array = _getArg("array", 0);
    int dim = _getArgAsInt("dim", 1);
    if (failed)
      return null;

    return arg_array.ubound(dim-1, block);
  }


  private Xobject _evalIntrinsic_shape()
  {
    Xobject arg_source = _getArg("source", 0);
    if (failed)
      return null;

    // not supported yet
    _error("cannot evaluate intrinsic function shape");
    return null;
  }


  private Xobject _evalIntrinsic_sizeWithoutDim()
  {
    Xobject arg_array = _getArg("array", 0);
    if (failed)
      return null;

    return _eval_sizeProduct(arg_array);
  }

  private Xobject _evalIntrinsic_sizeWithDim()
  {
    Xobject arg_array = _getArg("array", 0);
    int dim = _getArgAsInt("dim", 1);
    if (failed)
      return null;

    return arg_array.extent(dim-1, block);
  }


  private Xobject _eval_sizeProduct(Xobject array)
  {
    int rank = array.getFrank();
    long sizeL = 1L;

    for (int i = 0; i < rank; i++) {
      Xobject ext1 = array.extent(i);
      if (ext1.isIntConstant()) {
        sizeL = sizeL * ext1.getInt();
        if (sizeL >= 0x1<<31) {
          _warn("integer operation overflow to get array size");
          return null;
        }
      } else {
        /* not int constant */
        return null;
      }
    }

    return Xcons.IntConstant((int)sizeL);
  }


  private Xobject _evalIntrinsic_lcobound()
  {
    Xobject arg_coarray = _getArg("coarray", 0);
    if (failed)
      return null;

    _error("cannot evaluate intrinsic function \'lcobound\'");
    return null;
  }


  private Xobject _evalIntrinsic_ucobound()
  {
    Xobject arg_coarray = _getArg("coarray", 0);
    if (failed)
      return null;

    _error("cannot evaluate intrinsic function \'ucobound\'");
    return null;
  }


  private Xobject _evalIntrinsic_kind()
  {
    Xobject arg_x = _getArg("x", 0);
    if (failed)
      return null;

    /*** I don't know which representation is reliable ***/

    _restrict("cannot evaluate intrinsic function \'kind\'");

    //////////////////////
    System.out.println("GACCHA KIND");
    //////////////////////

    return null;
  }



  /*************************************************\
   * elemental intrinsic functions
  \*************************************************/
  private Xobject evalAsElementalIntrinsicFunction()
  {
    if (expr.Opcode() != Xcode.FUNCTION_CALL)
      return null;

    XobjList argList = (XobjList)expr.right();
    int nargs = 0;
    if (argList != null)
      nargs = argList.Nargs();

    String fname = expr.left().getName();
    fname = fname.toLowerCase();
    
    if (fname == "abs" && nargs == 1)
      return evalIntrinsic_abs();
    else if (fname == "dim" && nargs == 1)
      return evalIntrinsic_dim();
    else if (fname == "max" && nargs == 2)
      return evalIntrinsic_max(nargs);
    else if (fname == "min")
      return evalIntrinsic_min(nargs);
    else if (fname == "mod" && nargs == 2)
      return evalIntrinsic_mod();
    else if (fname == "modulo" && nargs == 2)
      return evalIntrinsic_modulo();
    else if (fname == "sign" && nargs == 1)
      return evalIntrinsic_sign();

    else if (fname == "len_trim" && nargs == 1)
      ;

    return null;
  }


  private Xobject evalIntrinsic_abs()
  {
    int val_a = _getArgAsInt("a", 0);
    if (failed)
      return null;

    int result = (val_a >= 0) ? val_a : - val_a;
    return Xcons.IntConstant(result);
  }

  private Xobject evalIntrinsic_dim()
  {
    int val_x = _getArgAsInt("x", 0);
    int val_y = _getArgAsInt("y", 1);
    if (failed)
      return null;

    int result = (val_x > val_y) ? val_x - val_y : 0;
    return Xcons.IntConstant(result);
  }


  private Xobject evalIntrinsic_max(int nargs)
  {
    if (nargs < 2)
      return null;

    int val = _getArgAsInt("a1", 0);
    if (failed)
      return null;

    for (int i = 2; i < nargs; i++) {
      int val1 = _getArgAsInt(null, i);
      if (failed)
        return null;
      if (val1 > val)
        val = val1;
    }

    return Xcons.IntConstant(val);
  }


  private Xobject evalIntrinsic_min(int nargs)
  {
    if (nargs < 2)
      return null;

    int val = _getArgAsInt("a1", 0);
    if (failed)
      return null;

    for (int i = 1; i < nargs; i++) {
      int val1 = _getArgAsInt(null, i);
      if (failed)
        return null;
      if (val1 < val)
        val = val1;
    }

    return Xcons.IntConstant(val);
  }


  private Xobject evalIntrinsic_mod()
  {
    int val_a = _getArgAsInt("a", 0);
    int val_p = _getArgAsInt("p", 1);
    if (failed)
      return null;

    if (val_p == 0) {
      _error("division by zero in intrinsic function \'mod\'");
      return null;
    }

    int result = val_a - (val_a / val_p) * val_p;
    return Xcons.IntConstant(result);
  }


  private Xobject evalIntrinsic_modulo()
  {
    int val_a = _getArgAsInt("a", 0);
    int val_p = _getArgAsInt("p", 1);
    if (failed)
      return null;

    if (val_p == 0) {
      _error("division by zero in intrinsic function \'modulo\'");
      return null;
    }

    int result = val_a - (val_a / val_p) * val_p;
    if (val_a < 0  && val_p > 0) {
      result = result + val_p;
    } else if (val_a > 0 && val_p < 0) {
      result = result - val_p;
    }
    return Xcons.IntConstant(result);
  }


  private Xobject evalIntrinsic_sign()
  {
    int val_a = _getArgAsInt("a", 0);
    int val_b = _getArgAsInt("b", 1);
    if (failed)
      return null;

    int result;
    if (val_b >= 0) 
      result = (val_a >= 0) ? val_a : - val_a;
    else
      result = (val_a <= 0) ? val_a : - val_a;
    return Xcons.IntConstant(result);
  }



  /*************************************************\
   * intrinsic operations
  \*************************************************/
  private Xobject evalAsIntrinsicUnaryOp(int arg1)
  {
    Xcode opcode = expr.Opcode();

    int result = 0;
    BigInteger bigResult = null;
    BigInteger bigArg1 = BigInteger.valueOf(arg1);
    boolean failed = false;

    switch (opcode) {
    case UNARY_MINUS_EXPR:
      result = - arg1;
      bigResult = bigArg1.negate();
      break;

  /*case UNARY_PLUS_EXPR:*/

  /*case UNARY_PAREN_EXPR:*/

    default:
      failed = true;
    }

    if (failed)
      return null;

    if (bigResult.compareTo(BigInteger.valueOf(result)) != 0)
      return null;

    return Xcons.IntConstant(result);
  }


  private Xobject evalAsIntrinsicBinaryOp(int arg1, int arg2)
  {
    Xcode opcode = expr.Opcode();

    int result = 0;
    BigInteger bigResult = null;
    BigInteger bigArg1 = BigInteger.valueOf(arg1);
    BigInteger bigArg2 = BigInteger.valueOf(arg2);
    boolean failed = false;

    switch (opcode) {
    case PLUS_EXPR:
      result = arg1 + arg2;
      bigResult = bigArg1.add(bigArg2);
      break;

    case MINUS_EXPR:
      result = arg1 - arg2;
      bigResult = bigArg1.subtract(bigArg2);
      break;

    case MUL_EXPR:
      result = arg1 * arg2;
      bigResult = bigArg1.multiply(bigArg2);
      break;

    case DIV_EXPR:    
      result = arg1 / arg2;
      bigResult = bigArg1.divide(bigArg2);
      break;

    case F_POWER_EXPR:
      failed = true;
      //result = Math.pow(arg1, arg2);
      //bigResult = bigArg1.pow(bigArg2);
      break;

    case LOG_NOT_EXPR:
    case LOG_EQ_EXPR:
    case LOG_NEQ_EXPR:
    case LOG_GE_EXPR:
    case LOG_GT_EXPR:
    case LOG_LE_EXPR:
    case LOG_LT_EXPR:
    case LOG_AND_EXPR:
    case LOG_OR_EXPR:
    case F_LOG_EQV_EXPR:
    case F_LOG_NEQV_EXPR:
      failed = true;
      break;
    }

    if (failed)
      return null;

    if (bigResult.compareTo(BigInteger.valueOf(result)) != 0)
      return null;

    return Xcons.IntConstant(result);
  }


  /*************************************************\
   * common parts
  \*************************************************/
  private int _getArgAsInt(String name, int pos)
  {
    Xobject arg = _getIntConstArg(name, pos);
    if (failed)
      return 0;

    return arg.getInt();
  }

  private Xobject _getIntConstArg(String name, int pos)
  {
    Xobject arg = _getArg(name, pos);
    if (failed)
      return null;

    arg = arg.cfold(block);     // do constant folding
    if (!arg.isIntConstant()) {
      failed = true;
      return null;
    }
    return arg;
  }


  private Xobject _getArg(String name, int pos)
  {
    Xobject arg;
    if (name != null)
      arg = argList.getArgWithKeyword(name, pos);
    else
      arg = argList.getArgOrNull(pos);

    if (arg == null) {
      failed = true;
      return null;
    }

    return arg;
  }


  /*************************************************\
   * error handler
  \*************************************************/
  private static void _info(String msg) {
    System.err.println("[XMP FconstFolder] INFO: " + msg);
  }

  private static void _warn(String msg) {
    System.err.println("[XMP FconstFolder] WARNING: " + msg);
  }

  private static void _error(String msg) {
    System.err.println("[XMP FconstFolder] ERROR: " + msg);
  }

  private static void _restrict(String msg) {
    System.err.println("[XMP FconstFolder] RESTRICTION: " + msg);
  }

}

