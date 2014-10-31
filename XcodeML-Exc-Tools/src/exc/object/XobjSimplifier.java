/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;
import exc.block.*;

/*******************************
 * This Object is not used so far.
 ***************************************/

public class XobjSimplifier
{
  private FuncDefBlock def;

  public XobjSimplifier()
  {
    def = null;
  }

  public XobjSimplifier(FuncDefBlock def)
  {
    this.def = def;
  }

  public Xobject run(Xobject expr)
  {
    /// default ///
    return expr;
  }

}