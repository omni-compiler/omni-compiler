/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

/**
 * Represents for statement in C, do statement in Fortran.
 */
public interface ForBlock
{
    /** canonicalize loop expressions */
    public void Canonicalize();
    /** whether or not canonicalized */
    public boolean isCanonical();
    /** get induction variable */
    public Xobject getInductionVar();
    /** get lower bound */
    public Xobject getLowerBound();
    /** set lower bound */
    public void setLowerBound(Xobject x);
    /** get upper bound */
    public Xobject getUpperBound();
    /** set upper bound */
    public void setUpperBound(Xobject x);
    /** get step */
    public Xobject getStep();
    /** set step */
    public void setStep(Xobject x);
    /** get initialization block */
    public BasicBlock getInitBBlock();
    /** get opcode of condition expression */
    public Xcode getCheckOpcode();
    /** get body */
    public BlockList getBody();
    /** translate to Xobject */
    public Xobject toXobject();
}
