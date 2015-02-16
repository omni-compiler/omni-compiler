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
 * information and methods about co-shape of Fortran coarray
 */
public class Coshape
{
  private final static int MAX_CORANK = 31;

  private Xobject[] codimensions;
  private int corank;

  /** constructor */
  public Coshape()
  {
    setCodimensions(new Xobject[0]);
  }
  public Coshape(Xobject[] codimensions)
  {
    setCodimensions(codimensions);
  }

  public int getCorank()
  {
    return corank;
  }

  public Xobject[] getCodimensions()
  {
    return codimensions;
  }

  public void setCodimensions(Xobject[] codimensions)
  {
    if (codimensions != null) {
      this.codimensions = codimensions;
      corank = codimensions.length;
    } else {   // non-coarray
      this.codimensions = new Xobject[0];
      corank = 0;
    }
  }

  public void removeCodimensions()
  {
    if (codimensions != null) {
      this.codimensions = new Xobject[0];
      corank = 0;
    }
  }

  public Xobject[] copyCodimensions()
  {
    Xobject[] codimensions1 = new Xobject[corank];
    // which is good?
    //    System.arraycopy(codimensions, 0, codimensions1, 0, corank);
    for (int i = 0; i < corank; i++)
      codimensions1[i] = codimensions[i].copy();
    return codimensions1;
  }

  public Coshape copy()
  {
    Xobject[] codim1 = copyCodimensions();
    return new Coshape(codim1);
  }
}

