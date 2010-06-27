/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import java.util.Set;

import xcodeml.c.binding.gen.IRVisitable;

/**
 * Implements the interface allows type objects to evaluate not in typeTable,
 * but first appeared in symbols/globalSymbols.
 */
public interface XcLazyEvalType
{
    /**
     * notify whether this {type, ident} is
     * needed to evaluate lazy.
     * (i.e. bindingS is not evaluated.
     *
     * @return true if type need to be lazy evaluated.
     */
    public boolean isLazyEvalType();

    /**
     * Sets whether is type need to be lazy evaluated.
     *
     * @param enable whether is type need to be lazy evaluated.
     */
    public void setIsLazyEvalType(boolean enable);

    /** 
     * return xcodeml bindings which is not evaluated.
     * objects which translated from these bindings 
     * must be child of this {type, ident}.
     *
     * @return xcodeml bindings which is not evaluated.
     */
    public IRVisitable[] getLazyBindings();

    /**
     * Sets xcodeml bindings need to be evaluated after.
     *
     * @param xcodeml bindings.
     */
    public void setLazyBindings(IRVisitable[] bindings);

    /**
     * Gets symbol name of variables which is used to define type.
     *
     * @return name of variables.
     */
    public Set<String> getDependVar();
}
