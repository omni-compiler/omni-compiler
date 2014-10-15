/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 * ID=060
 */

package exc.object;
import static xcodeml.util.XmLog.fatal;

/**
 * information and methods about Fortran coarray
 */
public class Fcoarray
{
  /** codimension expressions */
  private Xobject[] codimensions;
  private int numCodimensions;

  /** constructor */
  public Fcoarray()
  {
    setCodimensions(new Xobject[0]);
  }
  public Fcoarray(Xobject[] codimensions)
  {
    setCodimensions(codimensions);
  }

  public void setCodimensions(Xobject[] codimensions)
  {
    if (codimensions != null) {
      this.codimensions = codimensions;
      numCodimensions = codimensions.length;
    } else {   // non-coarray
      this.codimensions = new Xobject[0];
      numCodimensions = 0;
    }
  }

  public int getNumCodimensions()
  {
    return numCodimensions;
  }

  public Xobject[] getCodimensions()
  {
    return codimensions;
  }

  public Xobject[] copyCodimensions()
  {
    Xobject[] codimensions1 = new Xobject[numCodimensions];
    System.arraycopy(codimensions, 0, codimensions1, 0, numCodimensions);
    return codimensions1;
  }

  public Fcoarray copy()
  {
    Xobject[] codim1 = copyCodimensions();
    return new Fcoarray(codim1);
  }
}

